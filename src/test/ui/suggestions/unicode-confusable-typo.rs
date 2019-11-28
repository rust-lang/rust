// run-rustfix
#![feature(non_ascii_idents)]

struct ℝ𝓊𝓈𝓉;

fn main() {
    let ü = Rust;
    //~^ ERROR cannot find value `Rust` in this scope
    let _ = u;
    //~^ ERROR cannot find value `u` in this scope
}
