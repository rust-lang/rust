#![crate_type="lib"]

struct S<N>;
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`
//~| HELP if you intended `N` to be a const parameter

// Ensure that we don't emit the const param suggestion here:
struct T<N: Copy>;
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`

type A<N> = ();
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`
//~| HELP if you intended `N` to be a const parameter

// Ensure that we don't emit the const param suggestion here:
type B<N: Copy> = ();
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`
type C<N: Sized> = ();
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`
type D<N: ?Sized> = ();
//~^ ERROR type parameter `N` is never used
//~| HELP consider removing `N`
//~| HELP if you intended `N` to be a const parameter
