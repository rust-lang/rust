use crate::spec::{LinkArgs, LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "hexagonv60".to_string();
    base.max_atomic_width = Some(32);
    // FIXME: HVX length defaults are per-CPU
    base.features = "-small-data,+hvx-length128b".to_string();

    base.crt_static_default = false;
    base.atomic_cas = true;
    base.has_rpath = true;
    base.linker_is_gnu = false;
    base.dynamic_linking = true;
    base.executables = true;

    base.pre_link_args = LinkArgs::new();
    base.post_link_args = LinkArgs::new();

    Target {
        llvm_target: "hexagon-unknown-linux-musl".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: concat!(
            "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32",
            ":32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32",
            ":32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048",
            ":2048:2048"
        )
        .to_string(),
        arch: "hexagon".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    }
}
