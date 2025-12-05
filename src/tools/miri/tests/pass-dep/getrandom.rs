// mac-os `getrandom_01` does some pointer shenanigans
//@compile-flags: -Zmiri-permissive-provenance
//@revisions: isolation no_isolation
//@[no_isolation]compile-flags: -Zmiri-disable-isolation

/// Test direct calls of getrandom 0.1, 0.2 and 0.3.
fn main() {
    let mut data = vec![0; 16];

    // Old Solaris had a different return type for `getrandom`, and old versions of the getrandom crate
    // used that signature, which Miri is not happy about.
    #[cfg(not(target_os = "solaris"))]
    getrandom_01::getrandom(&mut data).unwrap();

    getrandom_02::getrandom(&mut data).unwrap();

    getrandom_03::fill(&mut data).unwrap();
}
