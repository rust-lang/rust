// test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/239
//@edition: 2024
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn foo<'a>() -> impl Send {
    if false {
        foo();
    }
    async {}
}

fn main() {}
