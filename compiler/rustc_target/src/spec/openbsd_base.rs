use crate::spec::{FramePointer, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "openbsd".to_string(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".to_string()],
        has_rpath: true,
        abi_return_struct_as_int: true,
        position_independent_executables: true,
        frame_pointer: FramePointer::Always, // FIXME 43575: should be MayOmit...
        relro_level: RelroLevel::Full,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
