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
#include "rust_rng.h"
#include "rust_util.h"

// Initialization helpers for ISAAC RNG

void
isaac_seed(rust_kernel* kernel, uint8_t* dest, size_t size) {
#ifdef __WIN32__
    HCRYPTPROV hProv;
    kernel->win32_require
        (_T("CryptAcquireContext"),
         CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT|CRYPT_SILENT));
    kernel->win32_require
        (_T("CryptGenRandom"), CryptGenRandom(hProv, size, (BYTE*) dest));
    kernel->win32_require
        (_T("CryptReleaseContext"), CryptReleaseContext(hProv, 0));
#else
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd == -1)
        kernel->fatal("error opening /dev/urandom: %s", strerror(errno));
    size_t amount = 0;
    do {
        ssize_t ret = read(fd, dest+amount, size-amount);
        if (ret < 0)
            kernel->fatal("error reading /dev/urandom: %s", strerror(errno));
        else if (ret == 0)
            kernel->fatal("somehow hit eof reading from /dev/urandom");
        amount += (size_t)ret;
    } while (amount < size);
    int ret = close(fd);
    // FIXME #3697: Why does this fail sometimes?
    if (ret != 0)
        kernel->log(log_warn, "error closing /dev/urandom: %s",
            strerror(errno));
#endif
}

void
isaac_init(rust_kernel *kernel, randctx *rctx, rust_vec_box* user_seed) {
    memset(rctx, 0, sizeof(randctx));

    char *env_seed = kernel->env->rust_seed;
    if (user_seed != NULL) {
        // ignore bytes after the required length
        size_t seed_len = user_seed->body.fill < sizeof(rctx->randrsl)
            ? user_seed->body.fill : sizeof(rctx->randrsl);
        memcpy(&rctx->randrsl, user_seed->body.data, seed_len);
    } else if (env_seed != NULL) {
        ub4 seed = (ub4) atoi(env_seed);
        for (size_t i = 0; i < RANDSIZ; i ++) {
            memcpy(&rctx->randrsl[i], &seed, sizeof(ub4));
            seed = (seed + 0x7ed55d16) + (seed << 12);
        }
    } else {
        isaac_seed(kernel, (uint8_t*) &rctx->randrsl, sizeof(rctx->randrsl));
    }

    randinit(rctx, 1);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
