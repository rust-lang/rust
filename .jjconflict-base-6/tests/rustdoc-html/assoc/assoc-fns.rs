// Basic testing for associated functions (in traits, trait impls & inherent impls).

//@ has assoc_fns/trait.Trait.html
pub trait Trait {
    //@ has - '//*[@id="tymethod.required"]' 'fn required(first: i32, second: &str)'
    fn required(first: i32, second: &str);

    //@ has - '//*[@id="method.provided"]' 'fn provided(only: ())'
    fn provided(only: ()) {}

    //@ has - '//*[@id="tymethod.params_are_unnamed"]' 'fn params_are_unnamed(_: i32, _: u32)'
    fn params_are_unnamed(_: i32, _: u32);
}
