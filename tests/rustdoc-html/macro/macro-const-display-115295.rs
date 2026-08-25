// https://github.com/rust-lang/rust/issues/115295
#![crate_name = "foo"]

//@ has foo/trait.Trait.html
pub trait Trait<T> {}

//@ has foo/struct.WithConst.html
pub struct WithConst<const N: usize>;

macro_rules! spans_from_macro {
    () => {
        impl WithConst<42> {
            pub fn new() -> Self {
                Self
            }
        }
        impl Trait<WithConst<42>> for WithConst<42> {}
        impl Trait<WithConst<43>> for WithConst<{ 43 }> {}
        impl Trait<WithConst<{ 44 }>> for WithConst<44> {}
        pub struct Other {
            pub field: WithConst<42>,
        }
    };
}

//@ has - '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl Trait<WithConst<41>> for WithConst<41>"
impl Trait<WithConst<41>> for WithConst<41> {}

//@ has - '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl WithConst<42>"
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl Trait<WithConst<42>> for WithConst<42>"
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl Trait<WithConst<43>> for WithConst<{ 43 }>"
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl Trait<WithConst<44>> for WithConst<44>"

//@ has foo/struct.Other.html
//@ has - //pre "pub field: WithConst<42>"
spans_from_macro!();
