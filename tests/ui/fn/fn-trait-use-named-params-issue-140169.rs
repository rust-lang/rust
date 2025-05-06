fn f1(_: fn(a: u8)) {}
fn f2(_: impl Fn(u8, vvvv: u8)) {} //~ ERROR `Trait(...)` syntax does not support named parameters
fn f3(_: impl Fn(aaaa: u8, u8)) {} //~ ERROR `Trait(...)` syntax does not support named parameters
fn f4(_: impl Fn(aaaa: u8, vvvv: u8)) {}
//~^ ERROR `Trait(...)` syntax does not support named parameters
//~| ERROR `Trait(...)` syntax does not support named parameters
fn f5(_: impl Fn(u8, ...)) {}
//~^ ERROR `Trait(...)` syntax does not support c_variadic parameters
fn f6(_: impl Fn(u8, #[allow(unused_attributes)] u8)) {}
//~^ ERROR `Trait(...)` syntax does not support attributes in parameters

fn main(){}
