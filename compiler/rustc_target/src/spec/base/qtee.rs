use crate::spec::targets::aarch64_unknown_none;
use crate::spec::{Os, PanicStrategy, RelocModel, RelroLevel, TargetOptions};

const ENTRY_POINT: &str = "tz_app_init";

pub(crate) fn opts() -> TargetOptions {
    let base = aarch64_unknown_none::target();

    TargetOptions {
        os: Os::Qtee,
        singlethread: true,
        relocation_model: RelocModel::Pic,
        dynamic_linking: true,
        only_cdylib: true,
        relro_level: RelroLevel::Full,
        has_thread_local: false,
        panic_strategy: PanicStrategy::Unwind,
        eh_frame_header: false,
        position_independent_executables: true,
        emit_debug_gdb_scripts: false,
        exe_suffix: ".rta".into(),
        entry_name: ENTRY_POINT.into(),
        ..base.options
    }
}
