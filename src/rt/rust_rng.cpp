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


#ifdef __WIN32__
void
win32_require(LPCTSTR fn, BOOL ok) {
    if (!ok) {
        LPTSTR buf;
        DWORD err = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                      FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf, 0, NULL );
        fprintf(stderr, "%s failed with error %ld: %s", fn, err, buf);
        LocalFree((HLOCAL)buf);
        abort();
    }
}
#endif

size_t
rng_seed_size() {
    randctx rctx;
    return sizeof(rctx.randrsl);
}

// Initialization helpers for ISAAC RNG

void
rng_gen_seed(uint8_t* dest, size_t size) {
#ifdef __WIN32__
    HCRYPTPROV hProv;
    win32_require
        (_T("CryptAcquireContext"),
         CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT|CRYPT_SILENT));
    win32_require
        (_T("CryptGenRandom"), CryptGenRandom(hProv, size, (BYTE*) dest));
    win32_require
        (_T("CryptReleaseContext"), CryptReleaseContext(hProv, 0));
#else
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "error opening /dev/urandom: %s", strerror(errno));
        abort();
    }
    size_t amount = 0;
    do {
        ssize_t ret = read(fd, dest+amount, size-amount);
        if (ret < 0) {
            fprintf(stderr, "error reading /dev/urandom: %s", strerror(errno));
            abort();
        }
        else if (ret == 0) {
            fprintf(stderr, "somehow hit eof reading from /dev/urandom");
            abort();
        }
        amount += (size_t)ret;
    } while (amount < size);
    int ret = close(fd);
    if (ret != 0) {
        fprintf(stderr, "error closing /dev/urandom: %s", strerror(errno));
        // FIXME #3697: Why does this fail sometimes?
        // abort();
    }
#endif
}

static void
isaac_init(randctx *rctx, char *env_seed,
           uint8_t* user_seed, size_t seed_len) {
    memset(rctx, 0, sizeof(randctx));

    if (user_seed != NULL) {
        // ignore bytes after the required length
        if (seed_len > sizeof(rctx->randrsl)) {
            seed_len = sizeof(rctx->randrsl);
        }
        memcpy(&rctx->randrsl, user_seed, seed_len);
    } else if (env_seed != NULL) {
        ub4 seed = (ub4) atoi(env_seed);
        for (size_t i = 0; i < RANDSIZ; i ++) {
            memcpy(&rctx->randrsl[i], &seed, sizeof(ub4));
            seed = (seed + 0x7ed55d16) + (seed << 12);
        }
    } else {
        rng_gen_seed((uint8_t*)&rctx->randrsl,
                     sizeof(rctx->randrsl));
    }

    randinit(rctx, 1);
}

void
rng_init(rust_rng* rng, char* env_seed,
         uint8_t *user_seed, size_t seed_len) {
    isaac_init(&rng->rctx, env_seed, user_seed, seed_len);
    rng->reseedable = !user_seed && !env_seed;
}

static void
rng_maybe_reseed(rust_rng* rng) {
    // If this RNG has generated more than 32KB of random data and was not
    // seeded by the user or RUST_SEED, then we should reseed now.
    const size_t RESEED_THRESHOLD = 32 * 1024;
    size_t bytes_generated = rng->rctx.randc * sizeof(ub4);
    if (bytes_generated < RESEED_THRESHOLD || !rng->reseedable) {
        return;
    }
    rng_gen_seed((uint8_t*)rng->rctx.randrsl,
                 sizeof(rng->rctx.randrsl));
    randinit(&rng->rctx, 1);
}

uint32_t
rng_gen_u32(rust_rng* rng) {
    uint32_t x = isaac_rand(&rng->rctx);
    rng_maybe_reseed(rng);
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
