// compile-pass

// Test that `dyn ... + ?Sized + ...` is okay (though `?Sized` has no effect in trait objects).

trait Foo {}

type _0 = dyn ?Sized + Foo;

type _1 = dyn Foo + ?Sized;

type _2 = dyn Foo + ?Sized + ?Sized;

type _3 = dyn ?Sized + Foo;

fn main() {}
