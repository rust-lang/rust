A non-constant value was used in a constant expression.

Erroneous code example:

```compile_fail,E0435
let foo = 42;
let a: [u8; foo]; // error: attempt to use a non-constant value in a constant
```

'constant' means 'a compile-time value'.

More details can be found in the [Variables and Mutability] section of the book.

[Variables and Mutability]: https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html#differences-between-variables-and-constants

To fix this error, please replace the value with a constant. Example:

```
let a: [u8; 42]; // ok!
```

Or:

```
const FOO: usize = 42;
let a: [u8; FOO]; // ok!
```
