//! Test the behaviour of `cfg(any())` and `cfg(all())`
#![allow(empty_cfg_predicate)]

#[cfg(any())]  // Equivalent to cfg(false)
struct Disabled;

#[cfg(all())]  // Equivalent to cfg(true)
struct Enabled;

fn main() {
    let _ = Disabled; //~ ERROR: cannot find value `Disabled`
    let _ = Enabled;  //  ok
}
