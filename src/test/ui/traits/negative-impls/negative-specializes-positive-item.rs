#![feature(specialization)]
#![feature(optin_builtin_traits)]

// Negative impl for u32 cannot "specialize" the base impl.
trait MyTrait {
    fn foo();
}
impl<T> MyTrait for T {
    default fn foo() {}
}
impl !MyTrait for u32 {} //~ ERROR E0748

fn main() {}
