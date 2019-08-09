// compile-flags: --error-format human-annotate-rs

pub fn main() {
    let x: Iter; //~ ERROR cannot find type `Iter` in this scope
}
