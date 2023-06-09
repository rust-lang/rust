// originally from ./tests/ui/pattern/usefulness/consts-opaque.rs
// panicked at 'assertion failed: rows.iter().all(|r| r.len() == v.len())',
// compiler/rustc_mir_build/src/thir/pattern/_match.rs:2030:5

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(PartialEq)]
struct Foo(i32);
const FOO_REF_REF: &&Foo = &&Foo(42);

fn main() {
    // This used to cause an ICE (https://github.com/rust-lang/rust/issues/78071)
    match FOO_REF_REF {
        FOO_REF_REF => {},
        Foo(_) => {},
    }
}
