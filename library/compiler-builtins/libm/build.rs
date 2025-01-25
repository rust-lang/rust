use std::env;

mod configure;

fn main() {
    let cfg = configure::Config::from_env();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(assert_no_panic)");

    let lvl = env::var("OPT_LEVEL").unwrap();
    if lvl != "0" && !cfg!(debug_assertions) {
        println!("cargo:rustc-cfg=assert_no_panic");
    } else if env::var("ENSURE_NO_PANIC").is_ok() {
        // Give us a defensive way of ensureing that no-panic is checked  when we
        // expect it to be.
        panic!("`assert_no_panic `was not enabled");
    }

    configure::emit_libm_config(&cfg);
}
