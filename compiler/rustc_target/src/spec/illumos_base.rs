use crate::spec::{FramePointer, LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut late_link_args = LinkArgs::new();
    late_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            // The illumos libc contains a stack unwinding implementation, as
            // does libgcc_s.  The latter implementation includes several
            // additional symbols that are not always in base libc.  To force
            // the consistent use of just one unwinder, we ensure libc appears
            // after libgcc_s in the NEEDED list for the resultant binary by
            // ignoring any attempts to add it as a dynamic dependency until the
            // very end.
            // FIXME: This should be replaced by a more complete and generic
            // mechanism for controlling the order of library arguments passed
            // to the linker.
            "-lc".to_string(),
            // LLVM will insert calls to the stack protector functions
            // "__stack_chk_fail" and "__stack_chk_guard" into code in native
            // object files.  Some platforms include these symbols directly in
            // libc, but at least historically these have been provided in
            // libssp.so on illumos and Solaris systems.
            "-lssp".to_string(),
        ],
    );

    TargetOptions {
        os: "illumos".to_string(),
        dynamic_linking: true,
        executables: true,
        has_rpath: true,
        families: vec!["unix".to_string()],
        is_like_solaris: true,
        linker_is_gnu: false,
        limit_rdylib_exports: false, // Linker doesn't support this
        frame_pointer: FramePointer::Always,
        eh_frame_header: false,
        late_link_args,

        // While we support ELF TLS, rust requires a way to register
        // cleanup handlers (in C, this would be something along the lines of:
        // void register_callback(void (*fn)(void *), void *arg);
        // (see src/libstd/sys/unix/fast_thread_local.rs) that is currently
        // missing in illumos.  For now at least, we must fallback to using
        // pthread_{get,set}specific.
        //has_thread_local: true,

        // FIXME: Currently, rust is invoking cc to link, which ends up
        // causing these to get included twice.  We should eventually transition
        // to having rustc invoke ld directly, in which case these will need to
        // be uncommented.
        //
        // We want XPG6 behavior from libc and libm.  See standards(5)
        //pre_link_objects_exe: vec![
        //    "/usr/lib/amd64/values-Xc.o".to_string(),
        //    "/usr/lib/amd64/values-xpg6.o".to_string(),
        //],
        ..Default::default()
    }
}
