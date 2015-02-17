% Whitespace [FIXME: needs RFC]

* Lines must not exceed 99 characters.
* Use 4 spaces for indentation, _not_ tabs.
* No trailing whitespace at the end of lines or files.

### Spaces

* Use spaces around binary operators, including the equals sign in attributes:

``` rust
#[deprecated = "Use `bar` instead."]
fn foo(a: uint, b: uint) -> uint {
    a + b
}
```

* Use a space after colons and commas:

``` rust
fn foo(a: Bar);

MyStruct { foo: 3, bar: 4 }

foo(bar, baz);
```

* Use a space after the opening and before the closing brace for
  single line blocks or `struct` expressions:

``` rust
spawn(proc() { do_something(); })

Point { x: 0.1, y: 0.3 }
```

### Line wrapping

* For multiline function signatures, each new line should align with the
  first parameter. Multiple parameters per line are permitted:

``` rust
fn frobnicate(a: Bar, b: Bar,
              c: Bar, d: Bar)
              -> Bar {
    ...
}

fn foo<T: This,
       U: That>(
       a: Bar,
       b: Bar)
       -> Baz {
    ...
}
```

* Multiline function invocations generally follow the same rule as for
  signatures. However, if the final argument begins a new block, the
  contents of the block may begin on a new line, indented one level:

``` rust
fn foo_bar(a: Bar, b: Bar,
           c: |Bar|) -> Bar {
    ...
}

// Same line is fine:
foo_bar(x, y, |z| { z.transpose(y) });

// Indented body on new line is also fine:
foo_bar(x, y, |z| {
    z.quux();
    z.rotate(x)
})
```

> **[FIXME]** Do we also want to allow the following?
>
> ```rust
> frobnicate(
>     arg1,
>     arg2,
>     arg3)
> ```
>
> This style could ease the conflict between line length and functions
> with many parameters (or long method chains).

### Matches

> * **[Deprecated]** If you have multiple patterns in a single `match`
>   arm, write each pattern on a separate line:
>
>     ``` rust
>     match foo {
>         bar(_)
>         | baz => quux,
>         x
>         | y
>         | z => {
>             quuux
>         }
>     }
>     ```

### Alignment

Idiomatic code should not use extra whitespace in the middle of a line
to provide alignment.


``` rust
// Good
struct Foo {
    short: f64,
    really_long: f64,
}

// Bad
struct Bar {
    short:       f64,
    really_long: f64,
}

// Good
let a = 0;
let radius = 7;

// Bad
let b        = 0;
let diameter = 7;
```
