#ifndef _FIX_WINSOCK2_H
#define _FIX_WINSOCK2_H 1

#include_next <winsock2.h>

typedef struct pollfd {
  SOCKET fd;
  short  events;
  short  revents;
} WSAPOLLFD, *PWSAPOLLFD, *LPWSAPOLLFD;

#endif
