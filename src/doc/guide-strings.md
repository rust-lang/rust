% The Strings Guide

Strings are an important concept to master in any programming language. If you
come from a managed language background, you may be surprised at the complexity
of string handling in a systems programming language. Efficient access and
allocation of memory for a dynamically sized structure involves a lot of
details. Luckily, Rust has lots of tools to help us here.

A **string** is a sequence of unicode scalar values encoded as a stream of
UTF-8 bytes. All strings are guaranteed to be validly-encoded UTF-8 sequences.
Additionally, strings are not null-terminated and can contain null bytes.

Rust has two main types of strings: `String` and `&str`. 

# String

A `String` is a heap-allocated string. Mutable strings are growable, 
and strings are guaranteed to be UTF-8. 


```{rust}
let mut s = "Hello".to_string();
println!("{}", s);

s.push_str(", world.");
println!("{}", s);
```

# &str

The second kind,`&str`, is pronounced a 'string slice'. A string slice is 
often used to represent a view in to an existing string. 

```{rust}
let s = "Hello world".to_string();
let hello: &str = s.as_slice().slice_to(5);
println!("{} world", hello);
```

String literals are also of the type `&str`, so the example can be rewritten as

```{rust}
let s: &str = "Hello world";
let hello: &str = s.slice_to(5);
println!("{} world", hello);
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
need, and it can make your lifetimes more complex. Furthermore, you can pass
either kind of string into `foo` by using `.as_slice()` on any `String` you
need to pass in, so the `&str` version is more flexible.

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

# Other Documentation

* [the `&str` API documentation](/std/str/index.html)
* [the `String` API documentation](std/string/index.html)
