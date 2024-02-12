// aux-build: issue-118809.rs

#[macro_use]
extern crate issue_118809;

#[derive(Deserialize)] //~ ERROR mismatched types [E0308]
pub struct Build {
}

fn main() {}
