// Regression test for issue #80233
// Tests that we don't ICE when processing auto traits
// https://github.com/rust-lang/rust/issues/80233

#![crate_type = "lib"]
#![crate_name = "foo"]
pub trait Trait1 {}

pub trait Trait2 {
    type Type2;
}

pub trait Trait3 {
    type Type3;
}

impl Trait2 for Struct1 {
    type Type2 = Struct1;
}

impl<I: Trait2> Trait2 for Vec<I> {
    type Type2 = Vec<I::Type2>;
}

impl<T: Trait1> Trait3 for T {
    type Type3 = Struct1;
}

impl<T: Trait3> Trait3 for Vec<T> {
    type Type3 = Vec<T::Type3>;
}

pub struct Struct1 {}

//@ has foo/struct.Question.html
//@ has - '//h3[@class="code-header"]' 'impl<T> Send for Question<T>'
pub struct Question<T: Trait1> {
    pub ins: <<Vec<T> as Trait3>::Type3 as Trait2>::Type2,
}
