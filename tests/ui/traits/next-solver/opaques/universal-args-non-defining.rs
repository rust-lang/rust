//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// The recursive call to `foo` results in the opaque type use `opaque<U, T> = ?unconstrained`.
// This needs to be supported and treated as a revealing use.

fn foo<T, U>(b: bool) -> impl Sized {
    if b {
        foo::<U, T>(b);
    }
    1u16
}

fn main() {}
