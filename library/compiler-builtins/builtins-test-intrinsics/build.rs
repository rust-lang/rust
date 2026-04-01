mod builtins_configure {
    include!("../compiler-builtins/configure.rs");
}

fn main() {
    println!("cargo::rerun-if-changed=../configure.rs");

    let cfg = builtins_configure::Config::from_env();
    builtins_configure::configure_aliases(&cfg);
}
