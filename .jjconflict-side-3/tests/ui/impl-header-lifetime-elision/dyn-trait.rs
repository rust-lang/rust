// Test that `impl MyTrait<'_> for &i32` is equivalent to `impl<'a,
// 'b> MyTrait<'a> for &'b i32`.

#![allow(warnings)]

use std::fmt::Debug;

// Equivalent to `Box<dyn Debug + 'static>`:
trait StaticTrait { }
impl StaticTrait for Box<dyn Debug> { }

// Equivalent to `Box<dyn Debug + 'static>`:
trait NotStaticTrait { }
impl NotStaticTrait for Box<dyn Debug + '_> { }

// Check that we don't err when the trait has a lifetime parameter.
trait TraitWithLifetime<'a> { }
impl NotStaticTrait for &dyn TraitWithLifetime<'_> { }

fn static_val<T: StaticTrait>(_: T) {
}

fn with_dyn_debug_static<'a>(x: Box<dyn Debug + 'a>) {
    static_val(x);
    //~^ ERROR borrowed data escapes outside of function
}

fn not_static_val<T: NotStaticTrait>(_: T) {
}

fn with_dyn_debug_not_static<'a>(x: Box<dyn Debug + 'a>) {
    not_static_val(x); // OK
}

fn main() {
}
