use crate::spec::targets::aarch64_unknown_none;
use crate::spec::{
    LinkOutputKind, LinkSelfContainedDefault, Os, PanicStrategy, RelocModel, RelroLevel, StaticCow,
    TargetOptions, crt_objects,
};

const ENTRY_POINT: &str = "tz_app_init";
const PRE_LINK_OBJECTS: &[&str] = &["applib.lib", "common_applib.o"];
const PRE_LINK_ARGS: &[StaticCow<str>] = &[
    StaticCow::Borrowed("-lcmnlib"),
    StaticCow::Borrowed("--export-dynamic"),
    StaticCow::Borrowed("--allow-shlib-undefined"),
];
const EXPORT_SYMBOLS: &[&str] = &[
    "CElfFile_invoke",
    "TA_APP_NAME",
    "TA_CUSTOM_PROPERTIES_NUM",
    "TA_MAX_STORAGE_FILES",
    "TA_METADATA",
    "__fini_array_end",
    "__fini_array_start",
    "__init_array_end",
    "__init_array_start",
    "__stack_chk_guard",
    "__stack_guard_size",
    "appCntxt",
    "data_section_base",
    "data_section_limit",
    "ta_acceptBufSize",
    "ta_cryptoSelfTest",
    "ta_customProperties",
    "ta_description",
    "ta_filesNoPersist",
    "ta_heapAddr",
    "ta_heapLimit",
    "ta_heapSize",
    "ta_minExecTimeout",
    "ta_minWaitTimeout",
    "ta_multiSession",
    "ta_stackAddr",
    "ta_stackLimit",
    "ta_stackSize",
    "ta_version",
    "text_section_base",
    "text_section_limit",
    "wrap_funcs_on_exit_ptr",
];

fn pre_link_objects_self_contained() -> crt_objects::CrtObjects {
    crt_objects::new(&[
        (LinkOutputKind::DynamicNoPicExe, PRE_LINK_OBJECTS),
        (LinkOutputKind::DynamicPicExe, PRE_LINK_OBJECTS),
        (LinkOutputKind::StaticNoPicExe, PRE_LINK_OBJECTS),
        (LinkOutputKind::StaticPicExe, PRE_LINK_OBJECTS),
        (LinkOutputKind::DynamicDylib, PRE_LINK_OBJECTS),
        (LinkOutputKind::StaticDylib, PRE_LINK_OBJECTS),
    ])
}

pub(crate) fn opts() -> TargetOptions {
    let base = aarch64_unknown_none::target();
    let mut pre_link_args = base.options.pre_link_args.clone();
    pre_link_args.values_mut().for_each(|v| v.extend_from_slice(PRE_LINK_ARGS));

    TargetOptions {
        os: Os::Qtee,
        singlethread: true,
        relocation_model: RelocModel::Pic,
        dynamic_linking: true,
        only_cdylib: true,
        relro_level: RelroLevel::Full,
        has_thread_local: false,
        panic_strategy: PanicStrategy::Unwind,
        link_self_contained: LinkSelfContainedDefault::True,
        pre_link_objects_self_contained: pre_link_objects_self_contained(),
        eh_frame_header: false,
        position_independent_executables: true,
        override_export_symbols: Some(
            EXPORT_SYMBOLS.iter().cloned().map(std::borrow::Cow::from).collect(),
        ),
        pre_link_args,
        emit_debug_gdb_scripts: false,
        exe_suffix: ".rta".into(),
        entry_name: ENTRY_POINT.into(),
        ..base.options
    }
}
