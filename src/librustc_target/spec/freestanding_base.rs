use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();

    args.insert(
        LinkerFlavor::Gcc,
        vec![
            // We want to be able to strip as much executable code as possible
            // from the linker command line, and this flag indicates to the
            // linker that it can avoid linking in dynamic libraries that don't
            // actually satisfy any symbols up to that point (as with many other
            // resolutions the linker does). This option only applies to all
            // following libraries so we're sure to pass it as one of the first
            // arguments.
            "-Wl,--as-needed".to_string(),
        ],
    );

    TargetOptions {
        dynamic_linking: false,
        executables: true,
        linker_is_gnu: true,
        has_rpath: false,
        pre_link_args: args,
        position_independent_executables: false,
        ..Default::default()
    }
}
