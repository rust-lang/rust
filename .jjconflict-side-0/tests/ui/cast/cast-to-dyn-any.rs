//! regression test for issue <https://github.com/rust-lang/rust/issues/22289>
fn main() {
    0 as &dyn std::any::Any; //~ ERROR non-primitive cast
}
