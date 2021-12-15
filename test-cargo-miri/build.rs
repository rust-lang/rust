use std::env;

#[cfg(miri)]
compile_error!("`miri` cfg should not be set in build script");

fn not_in_miri() -> i32 {
    // Inline assembly definitely does not work in Miri.
    let mut dummy = 42;
    unsafe {
        std::arch::asm!("/* {} */", in(reg) &mut dummy);
    }
    return dummy;
}

fn main() {
    not_in_miri();
    // Cargo calls `miri --print=cfg` to populate the `CARGO_CFG_*` env vars.
    // Make sure that the "miri" flag is set.
    assert!(env::var_os("CARGO_CFG_MIRI").is_some());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MIRITESTVAR");
    println!("cargo:rustc-env=MIRITESTVAR=testval");
}
