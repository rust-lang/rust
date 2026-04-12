// Issue: https://github.com/rust-lang/rust/issues/150103
// ICE when using `::` at start of nested imports
// caused by `{{root}}` appearing in diagnostic suggestions

mod A {
    use Iuse::{ ::Fish }; //~ ERROR unresolved import `Iuse`
}

mod B {
    use A::{::Fish}; //~ ERROR unresolved import `A::Fish`
}

fn main() {}
