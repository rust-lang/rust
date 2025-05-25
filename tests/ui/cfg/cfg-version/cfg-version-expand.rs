//@ run-pass
//@ rustc-env:RUSTC_OVERRIDE_VERSION_STRING=1.50.3

#![feature(cfg_version)]

#[cfg(version("1.49.0"))]
const ON_1_49_0: bool = true;
#[cfg(version("1.50"))]
const ON_1_50_0: bool = true;
#[cfg(not(version("1.51")))]
const ON_1_51_0: bool = false;

// This one uses the wrong syntax, so doesn't eval to true
#[warn(unexpected_cfgs)]
#[cfg(not(version = "1.48.0"))] //~ WARN unexpected `cfg` condition name: `version`
const ON_1_48_0: bool = false;

fn main() {
    assert!(!ON_1_48_0);
    assert!(ON_1_49_0);
    assert!(ON_1_50_0);
    assert!(!ON_1_51_0);
    assert!(cfg!(version("1.1")));
    assert!(cfg!(version("1.49")));
    assert!(cfg!(version("1.50.0")));
    assert!(cfg!(version("1.50.3")));
    assert!(!cfg!(version("1.50.4")));
    assert!(!cfg!(version("1.51")));
    assert!(!cfg!(version("1.100")));
}
