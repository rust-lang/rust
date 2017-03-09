# Unsized Types

Most types have a particular size, in bytes, that is knowable at compile time.
For example, an `i32` is thirty-two bits big, or four bytes. However, there are
some types which are useful to express, but do not have a defined size. These are
called ‘unsized’ or ‘dynamically sized’ types. One example is `[T]`. This type
represents a certain number of `T` in sequence. But we don’t know how many
there are, so the size is not known.

Rust understands a few of these types, but they have some restrictions. There
are three:

1. We can only manipulate an instance of an unsized type via a pointer. An
   `&[T]` works fine, but a `[T]` does not.
2. Variables and arguments cannot have dynamically sized types.
3. Only the last field in a `struct` may have a dynamically sized type; the
   other fields must not. Enum variants must not have dynamically sized types as
   data.

So why bother? Well, because `[T]` can only be used behind a pointer, if we
didn’t have language support for unsized types, it would be impossible to write
this:

```rust,ignore
impl Foo for str {
```

or

```rust,ignore
impl<T> Foo for [T] {
```

Instead, you would have to write:

```rust,ignore
impl Foo for &str {
```

Meaning, this implementation would only work for [references][ref], and not
other types of pointers. With the `impl for str`, all pointers, including (at
some point, there are some bugs to fix first) user-defined custom smart
pointers, can use this `impl`.

[ref]: references-and-borrowing.html

# ?Sized

If you want to write a function that accepts a dynamically sized type, you
can use the special bound syntax, `?Sized`:

```rust
struct Foo<T: ?Sized> {
    f: T,
}
```

This `?Sized`, read as “T may or may not be `Sized`”, which allows us to match
both sized and unsized types. All generic type parameters implicitly
have the `Sized` bound, so the `?Sized` can be used to opt-out of the implicit
bound.
