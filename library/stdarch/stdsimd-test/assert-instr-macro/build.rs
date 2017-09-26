use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let opt_level = env::var("OPT_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let profile = env::var("PROFILE").unwrap_or(String::new());
    if profile == "release" || opt_level >= 2 {
        println!("cargo:rustc-cfg=optimized");
    }
}
