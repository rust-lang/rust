use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![
        // We want to be able to strip as much executable code as possible
        // from the linker command line, and this flag indicates to the
        // linker that it can avoid linking in dynamic libraries that don't
        // actually satisfy any symbols up to that point (as with many other
        // resolutions the linker does). This option only applies to all
        // following libraries so we're sure to pass it as one of the first
        // arguments.
        "-Wl,--as-needed".to_string(),

        // Always enable NX protection when it is available
        "-Wl,-z,noexecstack".to_string()
    ]);

    TargetOptions {
        pre_link_args: args,
        executables: true,
        relocation_model: "static".to_string(),
        disable_redzone: true,
        eliminate_frame_pointer: false,
        target_family: None,
        linker_is_gnu: true,
        has_elf_tls: true,
        .. Default::default()
    }
}
