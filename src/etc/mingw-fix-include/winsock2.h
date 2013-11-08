#ifndef _FIX_WINSOCK2_H
#define _FIX_WINSOCK2_H 1

#include_next <winsock2.h>

// mingw 4.0.x has broken headers (#9246) but mingw-w64 does not.
#if defined(__MINGW_MAJOR_VERSION) && __MINGW_MAJOR_VERSION == 4

typedef struct pollfd {
  SOCKET fd;
  short  events;
  short  revents;
} WSAPOLLFD, *PWSAPOLLFD, *LPWSAPOLLFD;

#endif

#endif // _FIX_WINSOCK2_H
