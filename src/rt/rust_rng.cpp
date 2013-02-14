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
rng_gen_seed(rust_kernel* kernel, uint8_t* dest, size_t size) {
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

static void
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
        rng_gen_seed(kernel, (uint8_t*)&rctx->randrsl, sizeof(rctx->randrsl));
    }

    randinit(rctx, 1);
}

void
rng_init(rust_kernel* kernel, rust_rng* rng, rust_vec_box* user_seed) {
    isaac_init(kernel, &rng->rctx, user_seed);
    rng->reseedable = !user_seed && !kernel->env->rust_seed;
}

static void
rng_maybe_reseed(rust_kernel* kernel, rust_rng* rng) {
    // If this RNG has generated more than 32KB of random data and was not
    // seeded by the user or RUST_SEED, then we should reseed now.
    const size_t RESEED_THRESHOLD = 32 * 1024;
    size_t bytes_generated = rng->rctx.randc * sizeof(ub4);
    if (bytes_generated < RESEED_THRESHOLD || !rng->reseedable) {
        return;
    }

    uint32_t new_seed[RANDSIZ];
    rng_gen_seed(kernel, (uint8_t*) new_seed, RANDSIZ * sizeof(uint32_t));

    // Stir new seed into PRNG's entropy pool.
    for (size_t i = 0; i < RANDSIZ; i++) {
        rng->rctx.randrsl[i] ^= new_seed[i];
    }

    randinit(&rng->rctx, 1);
}

uint32_t
rng_gen_u32(rust_kernel* kernel, rust_rng* rng) {
    uint32_t x = isaac_rand(&rng->rctx);
    rng_maybe_reseed(kernel, rng);
    return x;
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
