{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "ConnectionError",
     "evalue": "HTTPConnectionPool(host='localhost', port=5000): Max retries exceeded with url: /results (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x000002C1D3E32AC8>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mConnectionRefusedError\u001b[0m                    Traceback (most recent call last)",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connection.py\u001b[0m in \u001b[0;36m_new_conn\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    158\u001b[0m             conn = connection.create_connection(\n\u001b[1;32m--> 159\u001b[1;33m                 (self._dns_host, self.port), self.timeout, **extra_kw)\n\u001b[0m\u001b[0;32m    160\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\util\\connection.py\u001b[0m in \u001b[0;36mcreate_connection\u001b[1;34m(address, timeout, source_address, socket_options)\u001b[0m\n\u001b[0;32m     79\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0merr\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 80\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     81\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\util\\connection.py\u001b[0m in \u001b[0;36mcreate_connection\u001b[1;34m(address, timeout, source_address, socket_options)\u001b[0m\n\u001b[0;32m     69\u001b[0m                 \u001b[0msock\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbind\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msource_address\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 70\u001b[1;33m             \u001b[0msock\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconnect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msa\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     71\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0msock\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mConnectionRefusedError\u001b[0m: [WinError 10061] No connection could be made because the target machine actively refused it",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mNewConnectionError\u001b[0m                        Traceback (most recent call last)",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connectionpool.py\u001b[0m in \u001b[0;36murlopen\u001b[1;34m(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)\u001b[0m\n\u001b[0;32m    599\u001b[0m                                                   \u001b[0mbody\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mbody\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mheaders\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 600\u001b[1;33m                                                   chunked=chunked)\n\u001b[0m\u001b[0;32m    601\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connectionpool.py\u001b[0m in \u001b[0;36m_make_request\u001b[1;34m(self, conn, method, url, timeout, chunked, **httplib_request_kw)\u001b[0m\n\u001b[0;32m    353\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 354\u001b[1;33m             \u001b[0mconn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mhttplib_request_kw\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    355\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\http\\client.py\u001b[0m in \u001b[0;36mrequest\u001b[1;34m(self, method, url, body, headers, encode_chunked)\u001b[0m\n\u001b[0;32m   1243\u001b[0m         \u001b[1;34m\"\"\"Send a complete request to the server.\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1244\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_send_request\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbody\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mencode_chunked\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1245\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\http\\client.py\u001b[0m in \u001b[0;36m_send_request\u001b[1;34m(self, method, url, body, headers, encode_chunked)\u001b[0m\n\u001b[0;32m   1289\u001b[0m             \u001b[0mbody\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_encode\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbody\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'body'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1290\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mendheaders\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbody\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mencode_chunked\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mencode_chunked\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1291\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\http\\client.py\u001b[0m in \u001b[0;36mendheaders\u001b[1;34m(self, message_body, encode_chunked)\u001b[0m\n\u001b[0;32m   1238\u001b[0m             \u001b[1;32mraise\u001b[0m \u001b[0mCannotSendHeader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1239\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_send_output\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmessage_body\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mencode_chunked\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mencode_chunked\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1240\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\http\\client.py\u001b[0m in \u001b[0;36m_send_output\u001b[1;34m(self, message_body, encode_chunked)\u001b[0m\n\u001b[0;32m   1025\u001b[0m         \u001b[1;32mdel\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_buffer\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1026\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmsg\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1027\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\http\\client.py\u001b[0m in \u001b[0;36msend\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m    965\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mauto_open\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 966\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconnect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    967\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connection.py\u001b[0m in \u001b[0;36mconnect\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    180\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mconnect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 181\u001b[1;33m         \u001b[0mconn\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_new_conn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    182\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_prepare_conn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mconn\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connection.py\u001b[0m in \u001b[0;36m_new_conn\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    167\u001b[0m             raise NewConnectionError(\n\u001b[1;32m--> 168\u001b[1;33m                 self, \"Failed to establish a new connection: %s\" % e)\n\u001b[0m\u001b[0;32m    169\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNewConnectionError\u001b[0m: <urllib3.connection.HTTPConnection object at 0x000002C1D3E32AC8>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mMaxRetryError\u001b[0m                             Traceback (most recent call last)",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\adapters.py\u001b[0m in \u001b[0;36msend\u001b[1;34m(self, request, stream, timeout, verify, cert, proxies)\u001b[0m\n\u001b[0;32m    448\u001b[0m                     \u001b[0mretries\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmax_retries\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 449\u001b[1;33m                     \u001b[0mtimeout\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    450\u001b[0m                 )\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\connectionpool.py\u001b[0m in \u001b[0;36murlopen\u001b[1;34m(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)\u001b[0m\n\u001b[0;32m    637\u001b[0m             retries = retries.increment(method, url, error=e, _pool=self,\n\u001b[1;32m--> 638\u001b[1;33m                                         _stacktrace=sys.exc_info()[2])\n\u001b[0m\u001b[0;32m    639\u001b[0m             \u001b[0mretries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msleep\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\urllib3\\util\\retry.py\u001b[0m in \u001b[0;36mincrement\u001b[1;34m(self, method, url, response, error, _pool, _stacktrace)\u001b[0m\n\u001b[0;32m    398\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mnew_retry\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_exhausted\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 399\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mMaxRetryError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m_pool\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0merror\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mResponseError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcause\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    400\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mMaxRetryError\u001b[0m: HTTPConnectionPool(host='localhost', port=5000): Max retries exceeded with url: /results (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x000002C1D3E32AC8>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mConnectionError\u001b[0m                           Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-2-04859c40d37c>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0murl\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'http://localhost:5000/results'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m \u001b[0mr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mrequests\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpost\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mjson\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m{\u001b[0m\u001b[1;34m'rate'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'sales_in_first_month'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m200\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'sales_in_second_month'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m400\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mr\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mjson\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\api.py\u001b[0m in \u001b[0;36mpost\u001b[1;34m(url, data, json, **kwargs)\u001b[0m\n\u001b[0;32m    114\u001b[0m     \"\"\"\n\u001b[0;32m    115\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 116\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mrequest\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'post'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mjson\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mjson\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    117\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    118\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\api.py\u001b[0m in \u001b[0;36mrequest\u001b[1;34m(method, url, **kwargs)\u001b[0m\n\u001b[0;32m     58\u001b[0m     \u001b[1;31m# cases, and look like a memory leak in others.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     59\u001b[0m     \u001b[1;32mwith\u001b[0m \u001b[0msessions\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSession\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0msession\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 60\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0msession\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0murl\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0murl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     61\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     62\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\sessions.py\u001b[0m in \u001b[0;36mrequest\u001b[1;34m(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)\u001b[0m\n\u001b[0;32m    531\u001b[0m         }\n\u001b[0;32m    532\u001b[0m         \u001b[0msend_kwargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msettings\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 533\u001b[1;33m         \u001b[0mresp\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprep\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0msend_kwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    534\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    535\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mresp\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\sessions.py\u001b[0m in \u001b[0;36msend\u001b[1;34m(self, request, **kwargs)\u001b[0m\n\u001b[0;32m    644\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    645\u001b[0m         \u001b[1;31m# Send the request\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 646\u001b[1;33m         \u001b[0mr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0madapter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    647\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    648\u001b[0m         \u001b[1;31m# Total elapsed time of the request (approximately)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\requests\\adapters.py\u001b[0m in \u001b[0;36msend\u001b[1;34m(self, request, stream, timeout, verify, cert, proxies)\u001b[0m\n\u001b[0;32m    514\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0mSSLError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrequest\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    515\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 516\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mConnectionError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrequest\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mrequest\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    517\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    518\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mClosedPoolError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mConnectionError\u001b[0m: HTTPConnectionPool(host='localhost', port=5000): Max retries exceeded with url: /results (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x000002C1D3E32AC8>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"
     ]
    }
   ],
   "source": [
    "import requests\n",
    "\n",
    "url = 'http://localhost:5000/results'\n",
    "r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})\n",
    "\n",
    "print(r.json())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
