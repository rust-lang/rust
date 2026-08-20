//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/248>

#![allow(warnings)]

fn foo() -> impl IntoIterator<Item = u32> {
    if false {
        let x: Vec<_> = foo().into_iter().collect();
    }

    [1, 2]
}

// `Flatten: Iterator` is ambiguous.
fn move_forward() -> impl IntoIterator<Item = i32> {
    std::iter::empty().map(|_: ()| move_forward()).flatten().collect::<Vec<_>>()
}

fn argument_types() -> impl IntoIterator<Item = i32> {
    argument_types().into_iter().collect::<Vec<_>>()
}

fn main() {}
