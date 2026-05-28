macro_rules! foo {
    ($id: ident) => {
        $id
    }
}

// Testing that the error span points to the parameter 'x' in the callsite,
// not to the macro variable '$id'
fn main() {
    foo!(
        x //~ ERROR cannot find value `x` in this scope
        );
}
