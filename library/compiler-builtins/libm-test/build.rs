#[path = "../../libm/configure.rs"]
mod configure;
use configure::Config;

fn main() {
    println!("cargo:rerun-if-changed=../../libm/configure.rs");
    let cfg = Config::from_env();
    configure::emit_test_config(&cfg);
}
