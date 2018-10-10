extern crate vergen;

use std::env;

fn main() {
    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
    // Don't rebuild miri even if nothing changed
    println!("cargo:rerun-if-changed=build.rs");
    // vergen
    vergen::generate_cargo_keys(vergen::ConstantsFlags::all())
        .expect("Unable to generate vergen keys!");
}
