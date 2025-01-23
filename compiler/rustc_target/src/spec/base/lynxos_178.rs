use std::borrow::Cow;
use std::collections::BTreeMap;
use std::env;

use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, RelroLevel, SplitDebuginfo, StackProbeType,
    TargetOptions, cvs,
};

/// Strip mutability from &mut str.
fn immutable(value: &mut str) -> &str {
    let immutable = value;
    immutable
}

macro_rules! cow_format {
    ($arg:expr) => {
        // Cow::from works with &str but not with &mut str.
        Cow::from(immutable(Box::leak(format!($arg).into_boxed_str())))
    };
}

pub(crate) fn opts() -> TargetOptions {
    let linker_flavor = LinkerFlavor::Gnu(Cc::No, Lld::No);
    let other_linker_flavor = LinkerFlavor::Gnu(Cc::No, Lld::Yes);
    let env_prefix = env::var("ENV_PREFIX").unwrap_or("ENV_PREFIX".into());
    let gcc_dir = env::var("GCC_DIR").unwrap_or("GCC_DIR".into());
    let cdk_path = format!("{env_prefix}/cdk/linux-elf-x86_64");

    let mut pre_link_args_v = vec![Cow::from("-m"), Cow::from("elf_x86_64_lynx178")];
    pre_link_args_v.push(cow_format!("-L{gcc_dir}"));
    pre_link_args_v.push(cow_format!("-L{env_prefix}/lib"));
    pre_link_args_v.push(cow_format!("-L{env_prefix}/usr/lib"));

    let mut post_link_args_v = Vec::new();
    post_link_args_v.push(cow_format!("{env_prefix}/lib/crt1.o"));
    post_link_args_v.push(cow_format!("{env_prefix}/lib/crti.o"));
    post_link_args_v.push(cow_format!("{gcc_dir}/crtbegin.o"));
    post_link_args_v.push(cow_format!("{gcc_dir}/crtend.o"));
    post_link_args_v.push(cow_format!("{env_prefix}/lib/crtn.o"));
    post_link_args_v.push(Cow::from("-lm"));
    post_link_args_v.push(Cow::from("-lgcc"));
    post_link_args_v.push(Cow::from("--start-group"));
    post_link_args_v.push(Cow::from("-lc"));
    post_link_args_v.push(Cow::from("-lpthread"));
    post_link_args_v.push(Cow::from("--end-group"));

    TargetOptions {
        os: "lynxos_178".into(),
        dynamic_linking: false,
        families: cvs!["unix"],
        position_independent_executables: false,
        static_position_independent_executables: false,
        relro_level: RelroLevel::Full,
        has_thread_local: false,
        crt_static_respected: true,
        panic_strategy: PanicStrategy::Abort,
        // Don't rely on the path to find the correct ld.
        linker: Some(cow_format!("{cdk_path}/bin/ld")),
        linker_flavor,
        eh_frame_header: false, // GNU ld (GNU Binutils) 2.37.50 does not support --eh-frame-hdr
        max_atomic_width: Some(64),
        pre_link_args: BTreeMap::from([
            (linker_flavor, pre_link_args_v.clone()),
            (other_linker_flavor, pre_link_args_v),
        ]),
        post_link_args: BTreeMap::from([
            (linker_flavor, post_link_args_v.clone()),
            (other_linker_flavor, post_link_args_v),
        ]),
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
