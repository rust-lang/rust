// run-pass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

// This tests that the `conservative_is_privately_uninhabited` fn doesn't cause
// ICEs by trying to evaluate `T::ASSOC` with an incorrect `ParamEnv`.

trait Foo {
    const ASSOC: usize = 1;
}

struct Iced<T: Foo>(T, [(); T::ASSOC])
where
    [(); T::ASSOC]: ;

impl Foo for u32 {}

fn main() {
    let _iced: Iced<u32> = return;
}
