#![warn(clippy::non_minimal_cfg)]

//~v non_minimal_cfg
#[cfg(all())]
fn all() {}

fn main() {}
