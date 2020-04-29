use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = super::windows_gnu_base::opts();

    // FIXME: This should be updated for the exception machinery changes from #67502
    // and inherit from `windows_gnu_base`, at least partially.
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
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,

        ..base
    }
}
