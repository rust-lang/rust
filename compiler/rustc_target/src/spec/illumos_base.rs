use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut late_link_args = LinkArgs::new();
    late_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
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
        os_family: Some("unix".to_string()),
        is_like_solaris: true,
        limit_rdylib_exports: false, // Linker doesn't support this
        eliminate_frame_pointer: false,
        eh_frame_header: false,
        late_link_args,

        // While we support ELF TLS, rust requires a way to register
        // cleanup handlers (in C, this would be something along the lines of:
        // void register_callback(void (*fn)(void *), void *arg);
        // (see src/libstd/sys/unix/fast_thread_local.rs) that is currently
        // missing in illumos.  For now at least, we must fallback to using
        // pthread_{get,set}specific.
        //has_elf_tls: true,

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
