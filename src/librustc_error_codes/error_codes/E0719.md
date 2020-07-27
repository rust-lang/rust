An associated type value was specified more than once.

Erroneous code example:

```compile_fail,E0719
#![feature(associated_type_bounds)]

trait FooTrait {}
trait BarTrait {}

// error: associated type `Item` in trait `Iterator` is specified twice
struct Foo<T: Iterator<Item: FooTrait, Item: BarTrait>> { f: T }
```

`Item` in trait `Iterator` cannot be specified multiple times for struct `Foo`.
To fix this, create a new trait that is a combination of the desired traits and
specify the associated type with the new trait.

Corrected example:

```
#![feature(associated_type_bounds)]

trait FooTrait {}
trait BarTrait {}
trait FooBarTrait: FooTrait + BarTrait {}

struct Foo<T: Iterator<Item: FooBarTrait>> { f: T } // ok!
```

For more information about associated types, see [the book][bk-at]. For more
information on associated type bounds, see [RFC 2289][rfc-2289].

[bk-at]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types
[rfc-2289]: https://rust-lang.github.io/rfcs/2289-associated-type-bounds.html
