extern crate vergen;

use std::env;

fn main() {
    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
    // Don't rebuild miri even if nothing changed
    println!("cargo:rerun-if-changed=build.rs");
    // vergen
    vergen().expect("Unable to generate vergen constants!");
}

fn vergen() -> vergen::Result<()> {
    use vergen::{ConstantsFlags, Vergen};

    let vergen = Vergen::new(ConstantsFlags::all())?;

    for (k, v) in vergen.build_info() {
        println!("cargo:rustc-env={}={}", k.name(), v);
    }

    Ok(())
}
