// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80
// characters.  In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use
// `:set tw=0` to disable.
syntax::register_diagnostics! {

E0178: r##"
In types, the `+` type operator has low precedence, so it is often necessary
to use parentheses.

For example:

```compile_fail,E0178
trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy,   // error, use &'a (Foo + Copy)
    x: &'a Foo + 'a,     // error, use &'a (Foo + 'a)
    y: &'a mut Foo + 'a, // error, use &'a mut (Foo + 'a)
    z: fn() -> Foo + 'a, // error, use fn() -> (Foo + 'a)
}
```

More details can be found in [RFC 438].

[RFC 438]: https://github.com/rust-lang/rfcs/pull/438
"##,

E0583: r##"
A file wasn't found for an out-of-line module.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
mod file_that_doesnt_exist; // error: file not found for module

fn main() {}
```

Please be sure that a file corresponding to the module exists. If you
want to use a module named `file_that_doesnt_exist`, you need to have a file
named `file_that_doesnt_exist.rs` or `file_that_doesnt_exist/mod.rs` in the
same directory.
"##,

E0584: r##"
A doc comment that is not attached to anything has been encountered.

Erroneous code example:

```compile_fail,E0584
trait Island {
    fn lost();

    /// I'm lost!
}
```

A little reminder: a doc comment has to be placed before the item it's supposed
to document. So if you want to document the `Island` trait, you need to put a
doc comment before it, not inside it. Same goes for the `lost` method: the doc
comment needs to be before it:

```
/// I'm THE island!
trait Island {
    /// I'm lost!
    fn lost();
}
```
"##,

E0585: r##"
A documentation comment that doesn't document anything was found.

Erroneous code example:

```compile_fail,E0585
fn main() {
    // The following doc comment will fail:
    /// This is a useless doc comment!
}
```

Documentation comments need to be followed by items, including functions,
types, modules, etc. Examples:

```
/// I'm documenting the following struct:
struct Foo;

/// I'm documenting the following function:
fn foo() {}
```
"##,

E0586: r##"
An inclusive range was used with no end.

Erroneous code example:

```compile_fail,E0586
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=]; // error: inclusive range was used with no end
}
```

An inclusive range needs an end in order to *include* it. If you just need a
start and no end, use a non-inclusive range (with `..`):

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..]; // ok!
}
```

Or put an end to your inclusive range:

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=3]; // ok!
}
```
"##,

E0704: r##"
This error indicates that a incorrect visibility restriction was specified.

Example of erroneous code:

```compile_fail,E0704
mod foo {
    pub(foo) struct Bar {
        x: i32
    }
}
```

To make struct `Bar` only visible in module `foo` the `in` keyword should be
used:
```
mod foo {
    pub(in crate::foo) struct Bar {
        x: i32
    }
}
# fn main() {}
```

For more information see the Rust Reference on [Visibility].

[Visibility]: https://doc.rust-lang.org/reference/visibility-and-privacy.html
"##,

E0743: r##"
C-variadic has been used on a non-foreign function.

Erroneous code example:

```compile_fail,E0743
fn foo2(x: u8, ...) {} // error!
```

Only foreign functions can use C-variadic (`...`). It is used to give an
undefined number of parameters to a given function (like `printf` in C). The
equivalent in Rust would be to use macros directly.
"##,

;

}
