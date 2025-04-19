#![allow(unexpected_cfgs)]

#[path = "../../libm/configure.rs"]
mod configure;

fn main() {
    println!("cargo:rerun-if-changed=../../libm/configure.rs");
    let cfg = configure::Config::from_env();
    configure::emit_libm_config(&cfg);
}
