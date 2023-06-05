## Types and Bounds

### Single line formatting

* `[T]` no spaces
* `[T; expr]`, e.g., `[u32; 42]`, `[Vec<Foo>; 10 * 2 + foo()]` (space after colon, no spaces around square brackets)
* `*const T`, `*mut T` (no space after `*`, space before type)
* `&'a T`, `&T`, `&'a mut T`, `&mut T` (no space after `&`, single spaces separating other words)
* `unsafe extern "C" fn<'a, 'b, 'c>(T, U, V) -> W` or `fn()` (single spaces around keywords and sigils, and after commas, no trailing commas, no spaces around brackets)
* `!` should be treated like any other type name, `Name`
* `(A, B, C, D)` (spaces after commas, no spaces around parens, no trailing comma unless it is a one-tuple)
* `<Baz<T> as SomeTrait>::Foo::Bar` or `Foo::Bar` or `::Foo::Bar` (no spaces around `::` or angle brackets, single spaces around `as`)
* `Foo::Bar<T, U, V>` (spaces after commas, no trailing comma, no spaces around angle brackets)
* `T + T + T` (single spaces between types, and `+`).
* `impl T + T + T` (single spaces between keyword, types, and `+`).

Parentheses used in types should not be surrounded by whitespace, e.g., `(Foo)`


### Line breaks

Avoid breaking lines in types where possible. Prefer breaking at outermost scope, e.g., prefer

```rust
Foo<
    Bar,
    Baz<Type1, Type2>,
>
```

to

```rust
Foo<Bar, Baz<
    Type1,
    Type2,
>>
```

`[T; expr]` may be broken after the `;` if necessary.

Function types may be broken following the rules for function declarations.

Generic types may be broken following the rules for generics.

Types with `+` may be broken after any `+` using block indent and breaking before the `+`. When breaking such a type, all `+`s should be line broken, e.g.,

```rust
impl Clone
    + Copy
    + Debug

Box<
    Clone
    + Copy
    + Debug
>
```
