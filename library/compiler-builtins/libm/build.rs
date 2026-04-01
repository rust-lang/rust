mod configure;

use configure::{emit_libm_config, env_flag, set_cfg};

fn main() {
    let cfg = configure::Config::from_env();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=configure.rs");
    println!("cargo:rustc-check-cfg=cfg(assert_no_panic)");

    // If set, enable `no-panic`. Requires LTO (`release-opt` profile).
    let no_panic = env_flag("ENSURE_NO_PANIC");
    set_cfg("assert_no_panic", no_panic);

    emit_libm_config(&cfg);
}
