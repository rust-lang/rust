int foo(void) {
    // Act as if using some API that's a lot newer than the deployment target.
    //
    // This forces Clang to insert a call to __isPlatformVersionAtLeast,
    // and linking will fail if that is not present.
    if (__builtin_available(
        macos 1000.0,
        ios 1000.0,
        tvos 1000.0,
        watchos 1000.0,
        // CI runs below Xcode 15, where `visionos` wasn't a valid key in
        // `__builtin_available`.
#ifdef TARGET_OS_VISION
        visionos 1000.0,
#endif
        *
    )) {
        return 1;
    } else {
        return 0;
    }
}
