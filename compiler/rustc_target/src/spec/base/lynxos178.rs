use std::borrow::Cow;

use crate::spec::{
    PanicStrategy, RelocModel, RelroLevel, SplitDebuginfo, StackProbeType, TargetOptions, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "lynxos178".into(),
        dynamic_linking: false,
        families: cvs!["unix"],
        position_independent_executables: false,
        static_position_independent_executables: false,
        relro_level: RelroLevel::Full,
        has_thread_local: false,
        crt_static_respected: true,
        panic_strategy: PanicStrategy::Abort,
        linker: Some(Cow::Borrowed("x86_64-lynx-lynxos178-gcc")),
        no_default_libraries: false,
        eh_frame_header: false, // GNU ld (GNU Binutils) 2.37.50 does not support --eh-frame-hdr
        max_atomic_width: Some(64),
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebuginfo::Packed,
            SplitDebuginfo::Unpacked,
            SplitDebuginfo::Off,
        ]),
        relocation_model: RelocModel::Static,
        stack_probes: StackProbeType::Inline,
        ..Default::default()
    }
}
