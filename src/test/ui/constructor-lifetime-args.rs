// All lifetime parameters in struct constructors are currently considered early bound,
// i.e., `S::<ARGS>` is interpreted kinda like an associated item `S::<ARGS>::ctor`.
// This behavior is a bit weird, because if equivalent constructor were written manually
// it would get late bound lifetime parameters.
// Variant constructors behave in the same way, lifetime parameters are considered
// belonging to the enum and being early bound.
// https://github.com/rust-lang/rust/issues/30904

struct S<'a, 'b>(&'a u8, &'b u8);
enum E<'a, 'b> {
    V(&'a u8),
    U(&'b u8),
}

fn main() {
    S(&0, &0); // OK
    S::<'static>(&0, &0);
    //~^ ERROR this struct takes 2 lifetime arguments but only 1 lifetime argument was supplied
    S::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR this struct takes 2 lifetime arguments but 3 lifetime arguments were supplied
    E::V(&0); // OK
    E::V::<'static>(&0);
    //~^ ERROR this enum takes 2 lifetime arguments but only 1 lifetime argument was supplied
    E::V::<'static, 'static, 'static>(&0);
    //~^ ERROR this enum takes 2 lifetime arguments but 3 lifetime arguments were supplied
}
