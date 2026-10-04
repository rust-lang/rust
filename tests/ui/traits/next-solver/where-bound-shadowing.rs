// regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/226

//@ check-pass
//@ compile-flags: -Znext-solver

fn into_param<T>(x: u32) -> (T, u64)
where
    u32: Into<T>,
{
    (x.into(), x.into())
}

fn main() {}
