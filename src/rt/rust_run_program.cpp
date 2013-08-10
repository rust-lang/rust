// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_globals.h"

#ifdef __APPLE__
#include <crt_externs.h>
#endif

#if defined(__WIN32__)

extern "C" CDECL void
rust_unset_sigprocmask() {
    // empty stub for windows to keep linker happy
}

extern "C" CDECL void
rust_set_environ(void* envp) {
    // empty stub for windows to keep linker happy
}

#elif defined(__GNUC__)

#include <signal.h>
#include <unistd.h>

#ifdef __FreeBSD__
extern char **environ;
#endif

extern "C" CDECL void
rust_unset_sigprocmask() {
    // this can't be safely converted to rust code because the
    // representation of sigset_t is platform-dependent
    sigset_t sset;
    sigemptyset(&sset);
    sigprocmask(SIG_SETMASK, &sset, NULL);
}

extern "C" CDECL void
rust_set_environ(void* envp) {
    // FIXME: this could actually be converted to rust (see issue #2674)
#ifdef __APPLE__
    *_NSGetEnviron() = (char **) envp;
#else
    environ = (char **) envp;
#endif
}

#else
#error "Platform not supported."
#endif

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
