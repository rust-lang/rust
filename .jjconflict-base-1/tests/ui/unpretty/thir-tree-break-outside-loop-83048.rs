// https://github.com/rust-lang/rust/issues/83048
//@ compile-flags: -Z unpretty=thir-tree

pub fn main() {
    break; //~ ERROR: `break` outside of a loop or labeled block [E0268]
}
