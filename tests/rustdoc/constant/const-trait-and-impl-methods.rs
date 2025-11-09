// check that we don't render assoc fns as `const` - even for const `trait`s and `impl`s.
#![crate_name = "foo"]
#![feature(const_trait_impl)]

//@ has foo/trait.Tr.html
//@ has - '//*[@id="tymethod.required"]' 'fn required()'
//@ !has - '//*[@id="tymethod.required"]' 'const'
//@ has - '//*[@id="method.defaulted"]' 'fn defaulted()'
//@ !has - '//*[@id="method.defaulted"]' 'const'
pub const trait Tr {
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
