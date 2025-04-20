int foo(void) {
    // Use some API that's a lot newer than the deployment target.
    // This forces Clang to insert a call to __isPlatformVersionAtLeast.
    if (__builtin_available(
        macos 1000.0,
        ios 1000.0,
        tvos 1000.0,
        watchos 1000.0,
        visionos 1000.0,
        *
    )) {
        return 1;
    } else {
        return 0;
    }
}
