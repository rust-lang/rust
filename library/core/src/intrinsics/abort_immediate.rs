cfg_select! {
    windows => {
        /// Ungracefully aborts the execution of the process.
        ///
        /// The stabilized version of this intrinsic is [`core::process::crash`].
        ///
        /// The fallback body is used unconditionally at runtime, but never in const or Miri.
        #[rustc_nounwind]
        #[rustc_intrinsic]
        #[rustc_do_not_const_check] // The body is not called in const
        #[inline(always)]
        pub const fn abort_immediate() -> ! {
            /// From `winnt.h`
            const FAST_FAIL_FATAL_APP_EXIT: crate::ffi::c_uint = 7;

            crate::os::windows::__fastfail(FAST_FAIL_FATAL_APP_EXIT)
        }
    }
    _ => {
        /// Ungracefully aborts the execution of the process.
        ///
        /// The stabilized version of this intrinsic is [`core::process::crash`].
        #[cfg(not(windows))]
        #[rustc_nounwind]
        #[rustc_intrinsic]
        pub const fn abort_immediate() -> !;
    }
}
