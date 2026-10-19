use std::env;

fn main() {
    // Forward the opt level so we can warn if the tests are going to be slow.
    let opt = env::var("OPT_LEVEL").expect("OPT_LEVEL unset");
    let profile = env::var("PROFILE").expect("PROFILE unset");
    println!("cargo::rustc-env=OPT_LEVEL={opt}");
    println!("cargo::rustc-env=PROFILE={profile}");
}
