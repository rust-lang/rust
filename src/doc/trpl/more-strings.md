% More Strings

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

The first kind is a `&str`. This is pronounced a 'string slice'.
String literals are of the type `&str`:

```
let string = "Hello there.";
```

Like any Rust reference, string slices have an associated lifetime. A string
literal is a `&'static str`.  A string slice can be written without an explicit
lifetime in many cases, such as in function arguments. In these cases the
lifetime will be inferred:

```
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}
```

Like vector slices, string slices are simply a pointer plus a length. This
means that they're a 'view' into an already-allocated string, such as a
string literal or a `String`.

# String

A `String` is a heap-allocated string. This string is growable, and is
also guaranteed to be UTF-8. `String`s are commonly created by
converting from a string slice using the `to_string` method.

```
let mut s = "Hello".to_string();
println!("{}", s);

s.push_str(", world.");
println!("{}", s);
```

A reference to a `String` will automatically coerce to a string slice:

```
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}

fn main() {
    let s = "Hello".to_string();
    takes_slice(&s);
}
```

You can also get a `&str` from a stack-allocated array of bytes:

```
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

if you have good reason. It's not polite to hold on to ownership you don't
need, and it can make your lifetimes more complex.

## Generic functions

To write a function that's generic over types of strings, use `&str`.

```
fn some_string_length(x: &str) -> uint {
    x.len()
}

fn main() {
    let s = "Hello, world";

    println!("{}", some_string_length(s));

    let s = "Hello, world".to_string();

    println!("{}", some_string_length(&s));
}
```

Both of these lines will print `12`.

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
through the characters anyway, which is an O(n) operation.

There's 3 basic levels of unicode (and its encodings):

- code units, the underlying data type used to store everything
- code points/unicode scalar values (char)
- graphemes (visible characters)

Rust provides iterators for each of these situations:

- `.bytes()` will iterate over the underlying bytes
- `.chars()` will iterate over the code points
- `.graphemes()` will iterate over each grapheme

Usually, the `graphemes()` method on `&str` is what you want:

```
# #![feature(unicode)]
let s = "u͔n͈̰̎i̙̮͚̦c͚̉o̼̩̰͗d͔̆̓ͥé";

for l in s.graphemes(true) {
    println!("{}", l);
}
```

This prints:

```text
u͔
n͈̰̎
i̙̮͚̦
c͚̉
o̼̩̰͗
d͔̆̓ͥ
é
```

Note that `l` has the type `&str` here, since a single grapheme can consist of
multiple codepoints, so a `char` wouldn't be appropriate.

This will print out each visible character in turn, as you'd expect: first `u͔`, then
`n͈̰̎`, etc. If you wanted each individual codepoint of each grapheme, you can use `.chars()`:

```
let s = "u͔n͈̰̎i̙̮͚̦c͚̉o̼̩̰͗d͔̆̓ͥé";

for l in s.chars() {
    println!("{}", l);
}
```

This prints:

```text
u
͔
n
̎
͈
̰
i
̙
̮
͚
̦
c
̉
͚
o
͗
̼
̩
̰
d
̆
̓
ͥ
͔
e
́
```

You can see how some of them are combining characters, and therefore the output
looks a bit odd.

If you want the individual byte representation of each codepoint, you can use
`.bytes()`:

```
let s = "u͔n͈̰̎i̙̮͚̦c͚̉o̼̩̰͗d͔̆̓ͥé";

for l in s.bytes() {
    println!("{}", l);
}
```

This will print:

```text
117
205
148
110
204
142
205
136
204
176
105
204
153
204
174
205
154
204
166
99
204
137
205
154
111
205
151
204
188
204
169
204
176
100
204
134
205
131
205
165
205
148
101
204
129
```

Many more bytes than graphemes!

# `Deref` coercions

References to `String`s will automatically coerce into `&str`s. Like this:

```
fn hello(s: &str) {
   println!("Hello, {}!", s);
}

let slice = "Steve";
let string = "Steve".to_string();

hello(slice);
hello(&string);
```
