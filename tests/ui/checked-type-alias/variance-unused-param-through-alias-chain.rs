// Check that we look through a chain of free alias types when working out which item is actually
// throwing an unused type parameter away. Free aliases aren't expanded by `type_of`, so naively
// walking the aliased type would just find `T` handed to the next alias and give up.

#![feature(checked_type_aliases)]

struct Ignores<T> {}
//~^ ERROR type parameter `T` is never used

type Forward<T> = Discard<T>;
type Discard<T> = Ignores<T>;

struct Wrap<T>(Forward<T>);
//~^ ERROR type parameter `T` is never used

// The same alias applied to different arguments is a different expansion. Treating `Id` as
// already-expanded after `Id<u8>` would hide the `T` in `Id<T>` and make `Repeat` look like it
// discards its parameter.
type Id<T> = T;
type Repeat<T> = (Id<u8>, Id<T>);

struct Recursive<T>(Repeat<Box<Recursive<T>>>);
//~^ ERROR type parameter `T` is only used recursively

fn main() {}
