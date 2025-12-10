//! Test the behaviour of `cfg(any())` and `cfg(all())`

#[cfg(any())]  // Equivalent to cfg(false)
struct Disabled;

#[cfg(all())]  // Equivalent to cfg(true)
struct Enabled;

fn main() {
    let _ = Disabled; //~ ERROR: cannot find value `Disabled`
    let _ = Enabled;  //  ok
}
