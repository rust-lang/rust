An unnecessary type or const parameter was given in a type alias.

Erroneous code example:

```compile_fail,E0091
type Foo<T> = u32; // error: type parameter `T` is unused
// or:
type Foo<A,B> = Box<A>; // error: type parameter `B` is unused
```

Please check you didn't write too many parameters. Example:

```
type Foo = u32; // ok!
type Foo2<A> = Box<A>; // ok!
```
