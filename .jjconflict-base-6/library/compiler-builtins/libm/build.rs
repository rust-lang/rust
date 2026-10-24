mod configure;

use configure::{Config, Library};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=configure.rs");

    let cfg = Config::from_env(Library::Libm);
    configure::emit(&cfg);
}
