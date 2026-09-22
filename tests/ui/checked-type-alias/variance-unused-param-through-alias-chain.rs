// A mention forwarded through free aliases still does not count as a use when the final type
// ignores the parameter. The diagnostic should label the original mention even though the
// intermediate aliases mention the parameter too.

#![feature(checked_type_aliases)]

struct Ignores<T> {}
//~^ ERROR type parameter `T` is never used

type Forward<T> = Discard<T>;
type Discard<T> = Ignores<T>;

struct Wrap<T>(Forward<T>);
//~^ ERROR type parameter `T` is never used

// An alias can occur with different arguments in the same type. Here `Repeat` preserves its
// parameter through `Id<T>`, so the enclosing struct uses `T` only recursively. The unrelated
// `Id<u8>` must not hide that recursive use.
type Id<T> = T;
type Repeat<T> = (Id<u8>, Id<T>);

struct Recursive<T>(Repeat<Box<Recursive<T>>>);
//~^ ERROR type parameter `T` is only used recursively

fn main() {}
