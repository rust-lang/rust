// check that we don't render `#[const_trait]` methods as `const` - even for
// const `trait`s and `impl`s.
#![crate_name = "foo"]
#![feature(const_trait_impl)]

//@ has foo/trait.Tr.html
//@ has - '//*[@id="tymethod.required"]' 'fn required()'
//@ !has - '//*[@id="tymethod.required"]' 'const'
//@ has - '//*[@id="method.defaulted"]' 'fn defaulted()'
//@ !has - '//*[@id="method.defaulted"]' 'const'
#[const_trait]
pub trait Tr {
    fn required();
    fn defaulted() {}
}

pub struct ConstImpl {}
pub struct NonConstImpl {}

//@ has foo/struct.ConstImpl.html
//@ has - '//*[@id="method.required"]' 'fn required()'
//@ !has - '//*[@id="method.required"]' 'const'
//@ has - '//*[@id="method.defaulted"]' 'fn defaulted()'
//@ !has - '//*[@id="method.defaulted"]' 'const'
impl const Tr for ConstImpl {
    fn required() {}
}

//@ has foo/struct.NonConstImpl.html
//@ has - '//*[@id="method.required"]' 'fn required()'
//@ !has - '//*[@id="method.required"]' 'const'
//@ has - '//*[@id="method.defaulted"]' 'fn defaulted()'
//@ !has - '//*[@id="method.defaulted"]' 'const'
impl Tr for NonConstImpl {
    fn required() {}
}
