// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
