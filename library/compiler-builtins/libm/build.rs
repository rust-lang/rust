use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(assert_no_panic)");
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"unstable\"))");

    println!("cargo::rustc-check-cfg=cfg(feature, values(\"checked\"))");

    #[allow(unexpected_cfgs)]
    if !cfg!(feature = "checked") {
        let lvl = env::var("OPT_LEVEL").unwrap();
        if lvl != "0" {
            println!("cargo:rustc-cfg=assert_no_panic");
        }
    }
}
