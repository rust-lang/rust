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
        dynamic_linking: true,
        executables: true,
        has_rpath: true,
        target_family: Some("unix".to_string()),
        is_like_solaris: true,
        limit_rdylib_exports: false, // Linker doesn't support this
        late_link_args,

        ..Default::default()
    }
}
