mod builtins_configure {
    include!("../compiler-builtins/configure.rs");
}

fn main() {
    println!("cargo::rerun-if-changed=../configure.rs");

    let target = builtins_configure::Target::from_env();
    builtins_configure::configure_f16_f128(&target);
    builtins_configure::configure_aliases(&target);
}
