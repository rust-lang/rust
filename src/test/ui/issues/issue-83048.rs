// compile-flags: -Z unpretty=thir-tree

pub fn main() {
    break; //~ ERROR: `break` outside of a loop [E0268]
}
