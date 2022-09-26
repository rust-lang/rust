use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "hexagonv60".into();
    base.max_atomic_width = Some(32);
    // FIXME: HVX length defaults are per-CPU
    base.features = "-small-data,+hvx-length128b".into();

    base.crt_static_default = false;
    base.has_rpath = true;
    base.linker_is_gnu = false;

    base.c_enum_min_bits = 8;

    Target {
        llvm_target: "hexagon-unknown-linux-musl".into(),
        pointer_width: 32,
        data_layout: concat!(
            "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32",
            ":32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32",
            ":32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048",
            ":2048:2048"
        )
        .into(),
        arch: "hexagon".into(),
        options: base,
    }
}
