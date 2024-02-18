//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// This tests that the inhabited check doesn't cause
// ICEs by trying to evaluate `T::ASSOC` with an incorrect `ParamEnv`.

trait Foo {
    const ASSOC: usize = 1;
}

#[allow(dead_code)]
struct Iced<T: Foo>(T, [(); T::ASSOC])
where
    [(); T::ASSOC]: ;

impl Foo for u32 {}

fn main() {
    let _iced: Iced<u32> = return;
}
