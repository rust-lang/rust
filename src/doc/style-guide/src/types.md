# Types and Bounds

## Single line formatting

- `[T]` no spaces
- `[T; expr]`, e.g., `[u32; 42]`, `[Vec<Foo>; 10 * 2 + foo()]` (space after colon, no spaces around square brackets)
- `*const T`, `*mut T` (no space after `*`, space before type)
- `&'a T`, `&T`, `&'a mut T`, `&mut T` (no space after `&`, single spaces separating other words)
- `unsafe extern "C" fn<'a, 'b, 'c>(T, U, V) -> W` or `fn()` (single spaces around keywords and sigils, and after commas, no trailing commas, no spaces around brackets)
- `!` gets treated like any other type name, `Name`
- `(A, B, C, D)` (spaces after commas, no spaces around parens, no trailing comma unless it is a one-tuple)
- `<Baz<T> as SomeTrait>::Foo::Bar` or `Foo::Bar` or `::Foo::Bar` (no spaces around `::` or angle brackets, single spaces around `as`)
- `Foo::Bar<T, U, V>` (spaces after commas, no trailing comma, no spaces around angle brackets)
- `T + T + T` (single spaces between types, and `+`).
- `impl T + T + T` (single spaces between keyword, types, and `+`).

Do not put space around parentheses used in types, e.g., `(Foo)`

## Line breaks

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

If a type requires line-breaks in order to fit, this section outlines where to
break such types if necessary.

Break `[T; expr]` after the `;` if necessary.

Break function types following the rules for function declarations.

Break generic types following the rules for generics.

Break types with `+` by breaking before the `+` and block-indenting the
subsequent lines. When breaking such a type, break before *every* `+`:

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

## Precise capturing bounds

A `use<'a, T>` precise capturing bound is formatted as if it were a single path segment with non-turbofished angle-bracketed args, like a trait bound whose identifier is `use`.

```rust
fn foo() -> impl Sized + use<'a> {}

// is formatted analogously to:

fn foo() -> impl Sized + Use<'a> {}
```
