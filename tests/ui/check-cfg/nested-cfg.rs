//@ check-pass

#[cfg(unknown)] //~ WARN unexpected `cfg` condition name
#[cfg(false)]
#[cfg(unknown)] // Should not warn
fn foo() {}

fn main() {}
