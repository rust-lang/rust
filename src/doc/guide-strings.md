% The Guide to Rust Strings

Strings are an important concept to master in any programming language. If you
come from a managed language background, you may be surprised at the complexity
of string handling in a systems programming language. Efficient access and
allocation of memory for a dynamically sized structure involves a lot of
details. Luckily, Rust has lots of tools to help us here.

A **string** is a sequence of unicode scalar values encoded as a stream of
UTF-8 bytes. All strings are guaranteed to be validly-encoded UTF-8 sequences.
Additionally, strings are not null-terminated and can contain null bytes.

Rust has two main types of strings: `&str` and `String`.

# &str

The first kind is a `&str`. This is pronounced a 'string slice.' String literals
are of the type `&str`:

```{rust}
let string = "Hello there.";
```

Like any Rust type, string slices have an associated lifetime. A string literal
is a `&'static str`.  A string slice can be written without an explicit
lifetime in many cases, such as in function arguments. In these cases the
lifetime will be inferred:

```{rust}
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}
```

Like vector slices, string slices are simply a pointer plus a length. This
means that they're a 'view' into an already-allocated string, such as a
`&'static str` or a `String`.

# String

A `String` is a heap-allocated string. This string is growable, and is also
guaranteed to be UTF-8.

```{rust}
let mut s = "Hello".to_string();
println!("{}", s);

s.push_str(", world.");
println!("{}", s);
```

You can coerce a `String` into a `&str` with the `as_slice()` method:

```{rust}
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}

fn main() {
    let s = "Hello".to_string();
    takes_slice(s.as_slice());
}
```

You can also get a `&str` from a stack-allocated array of bytes:

```{rust}
use std::str;

let x: &[u8] = &[b'a', b'b'];
let stack_str: &str = str::from_utf8(x).unwrap();
```

# Best Practices

## `String` vs. `&str`

In general, you should prefer `String` when you need ownership, and `&str` when
you just need to borrow a string. This is very similar to using `Vec<T>` vs. `&[T]`,
and `T` vs `&T` in general.

This means starting off with this:

```{rust,ignore}
fn foo(s: &str) {
```

and only moving to this:

```{rust,ignore}
fn foo(s: String) {
```

If you have good reason. It's not polite to hold on to ownership you don't
need, and it can make your lifetimes more complex.

## Generic functions

To write a function that's generic over types of strings, use [the `Str`
trait](http://doc.rust-lang.org/std/str/trait.Str.html):

```{rust}
fn some_string_length<T: Str>(x: T) -> uint {
        x.as_slice().len()
}

fn main() {
    let s = "Hello, world";

    println!("{}", some_string_length(s));

    let s = "Hello, world".to_string();

    println!("{}", some_string_length(s));
}
```

Both of these lines will print `12`. 

The only method that the `Str` trait has is `as_slice()`, which gives you
access to a `&str` value from the underlying string.

## Comparisons

To compare a String to a constant string, prefer `as_slice()`...

```{rust}
fn compare(string: String) {
    if string.as_slice() == "Hello" {
        println!("yes");
    }
}
```

... over `to_string()`:

```{rust}
fn compare(string: String) {
    if string == "Hello".to_string() {
        println!("yes");
    }
}
```

Converting a `String` to a `&str` is cheap, but converting the `&str` to a
`String` involves an allocation.

## Indexing strings

You may be tempted to try to access a certain character of a `String`, like
this:

```{rust,ignore}
let s = "hello".to_string();

println!("{}", s[0]);
```

This does not compile. This is on purpose. In the world of UTF-8, direct
indexing is basically never what you want to do. The reason is that each
character can be a variable number of bytes. This means that you have to iterate
through the characters anyway, which is a O(n) operation. 

To iterate over a string, use the `graphemes()` method on `&str`:

```{rust}
let s = "αἰθήρ";

for l in s.graphemes(true) {
    println!("{}", l);
}
```

Note that `l` has the type `&str` here, since a single grapheme can consist of
multiple codepoints, so a `char` wouldn't be appropriate.

This will print out each character in turn, as you'd expect: first "α", then
"ἰ", etc. You can see that this is different than just the individual bytes.
Here's a version that prints out each byte:

```{rust}
let s = "αἰθήρ";

for l in s.bytes() {
    println!("{}", l);
}
```

This will print:

```{notrust,ignore}
206
177
225
188
176
206
184
206
174
207
129
```

Many more bytes than graphemes!

# Other Documentation

* [the `&str` API documentation](std/str/index.html)
* [the `String` API documentation](std/string/index.html)
