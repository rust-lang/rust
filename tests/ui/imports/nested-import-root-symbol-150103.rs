// Issue: https://github.com/rust-lang/rust/issues/150103
// ICE when using `::` at start of nested imports
// caused by `{{root}}` appearing in diagnostic suggestions

mod A {
    use Iuse::{ ::Fish }; //~ ERROR failed to resolve: use of unresolved module or unlinked crate
}

mod B {
    use A::{::Fish}; //~ ERROR failed to resolve: crate root in paths can only be used in start position
}

fn main() {}
