type T = dyn core::fmt::Debug;
//~^ NOTE this type alias is unsized

fn f() -> T { loop {} }
//~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
//~| HELP the trait `Sized` is not implemented for `(dyn Debug + 'static)`
//~| NOTE doesn't have a size known at compile-time
//~| NOTE the return type of a function must have a statically known size

fn main() {}
