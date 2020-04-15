use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = super::windows_gnu_base::opts();

    // FIXME: Consider adding `-nostdlib` and inheriting from `windows_gnu_base`.
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            // Tell GCC to avoid linker plugins, because we are not bundling
            // them with Windows installer, and Rust does its own LTO anyways.
            "-fno-use-linker-plugin".to_string(),
            // Always enable DEP (NX bit) when it is available
            "-Wl,--nxcompat".to_string(),
        ],
    );

    // FIXME: This should be updated for the exception machinery changes from #67502.
    let mut late_link_args = LinkArgs::new();
    let late_link_args_dynamic = LinkArgs::new();
    let late_link_args_static = LinkArgs::new();
    late_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            //"-lwinstorecompat".to_string(),
            //"-lmingwex".to_string(),
            //"-lwinstorecompat".to_string(),
            "-lwinstorecompat".to_string(),
            "-lruntimeobject".to_string(),
            "-lsynchronization".to_string(),
            "-lvcruntime140_app".to_string(),
            "-lucrt".to_string(),
            "-lwindowsapp".to_string(),
            "-lmingwex".to_string(),
            "-lmingw32".to_string(),
        ],
    );

    TargetOptions {
        executables: false,
        limit_rdylib_exports: false,
        pre_link_args,
        // FIXME: Consider adding `-nostdlib` and inheriting from `windows_gnu_base`.
        pre_link_objects_exe: vec!["rsbegin.o".to_string()],
        // FIXME: Consider adding `-nostdlib` and inheriting from `windows_gnu_base`.
        pre_link_objects_dll: vec!["rsbegin.o".to_string()],
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,

        ..base
    }
}
