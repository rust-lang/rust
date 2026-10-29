#[path = "../libm/configure.rs"]
mod configure;

use configure::{Config, Library};

fn main() {
    println!("cargo:rerun-if-changed=../libm/configure.rs");
    let cfg = Config::from_env(Library::LibmTest);
    configure::emit(&cfg);
}
