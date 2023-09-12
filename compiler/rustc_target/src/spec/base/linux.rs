use crate::spec::{cvs, RelroLevel, SplitDebuginfo, TargetOptions};
use std::borrow::Cow;

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "linux".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_respected: true,
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebuginfo::Packed,
            SplitDebuginfo::Unpacked,
            SplitDebuginfo::Off,
        ]),
        ..Default::default()
    }
}
