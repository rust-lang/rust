#![allow(unexpected_cfgs)]

#[path = "../../configure.rs"]
mod configure;

fn main() {
    println!("cargo:rerun-if-changed=../../configure.rs");
    let cfg = configure::Config::from_env();
    configure::emit_libm_config(&cfg);
}
