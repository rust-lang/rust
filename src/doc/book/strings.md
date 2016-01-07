% Strings

Strings are an important concept for any programmer to master. Rust’s string
handling system is a bit different from other languages, due to its systems
focus. Any time you have a data structure of variable size, things can get
tricky, and strings are a re-sizable data structure. That being said, Rust’s
strings also work differently than in some other systems languages, such as C.

Let’s dig into the details. A ‘string’ is a sequence of Unicode scalar values
encoded as a stream of UTF-8 bytes. All strings are guaranteed to be a valid
encoding of UTF-8 sequences. Additionally, unlike some systems languages,
strings are not null-terminated and can contain null bytes.

Rust has two main types of strings: `&str` and `String`. Let’s talk about
`&str` first. These are called ‘string slices’. A string slice has a fixed
size, and cannot be mutated. It is a reference to a sequence of UTF-8 bytes.

```rust
let greeting = "Hello there."; // greeting: &'static str
```

`"Hello there."` is a string literal and its type is `&'static str`. A string
literal is a string slice that is statically allocated, meaning that it’s saved
inside our compiled program, and exists for the entire duration it runs. The
`greeting` binding is a reference to this statically allocated string. Any
function expecting a string slice will also accept a string literal.

String literals can span multiple lines. There are two forms. The first will
include the newline and the leading spaces:

```rust
let s = "foo
    bar";

assert_eq!("foo\n        bar", s);
```

The second, with a `\`, trims the spaces and the newline:

```rust
let s = "foo\
    bar"; 

assert_eq!("foobar", s);
```

Rust has more than only `&str`s though. A `String`, is a heap-allocated string.
This string is growable, and is also guaranteed to be UTF-8. `String`s are
commonly created by converting from a string slice using the `to_string`
method.

```rust
let mut s = "Hello".to_string(); // mut s: String
println!("{}", s);

s.push_str(", world.");
println!("{}", s);
```

`String`s will coerce into `&str` with an `&`:

```rust
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}

fn main() {
    let s = "Hello".to_string();
    takes_slice(&s);
}
```

This coercion does not happen for functions that accept one of `&str`’s traits
instead of `&str`. For example, [`TcpStream::connect`][connect] has a parameter
of type `ToSocketAddrs`. A `&str` is okay but a `String` must be explicitly
converted using `&*`.

```rust,no_run
use std::net::TcpStream;

TcpStream::connect("192.168.0.1:3000"); // &str parameter

let addr_string = "192.168.0.1:3000".to_string();
TcpStream::connect(&*addr_string); // convert addr_string to &str
```

Viewing a `String` as a `&str` is cheap, but converting the `&str` to a
`String` involves allocating memory. No reason to do that unless you have to!

## Indexing

Because strings are valid UTF-8, strings do not support indexing:

```rust,ignore
let s = "hello";

println!("The first letter of s is {}", s[0]); // ERROR!!!
```

Usually, access to a vector with `[]` is very fast. But, because each character
in a UTF-8 encoded string can be multiple bytes, you have to walk over the
string to find the nᵗʰ letter of a string. This is a significantly more
expensive operation, and we don’t want to be misleading. Furthermore, ‘letter’
isn’t something defined in Unicode, exactly. We can choose to look at a string as
individual bytes, or as codepoints:

```rust
let hachiko = "忠犬ハチ公";

for b in hachiko.as_bytes() {
    print!("{}, ", b);
}

println!("");

for c in hachiko.chars() {
    print!("{}, ", c);
}

println!("");
```

This prints:

```text
229, 191, 160, 231, 138, 172, 227, 131, 143, 227, 131, 129, 229, 133, 172,
忠, 犬, ハ, チ, 公,
```

As you can see, there are more bytes than `char`s.

You can get something similar to an index like this:

```rust
# let hachiko = "忠犬ハチ公";
let dog = hachiko.chars().nth(1); // kinda like hachiko[1]
```

This emphasizes that we have to walk from the beginning of the list of `chars`.

## Slicing

You can get a slice of a string with slicing syntax:

```rust
let dog = "hachiko";
let hachi = &dog[0..5];
```

But note that these are _byte_ offsets, not _character_ offsets. So
this will fail at runtime:

```rust,should_panic
let dog = "忠犬ハチ公";
let hachi = &dog[0..2];
```

with this error:

```text
thread '<main>' panicked at 'index 0 and/or 2 in `忠犬ハチ公` do not lie on
character boundary'
```

## Concatenation

If you have a `String`, you can concatenate a `&str` to the end of it:

```rust
let hello = "Hello ".to_string();
let world = "world!";

let hello_world = hello + world;
```

But if you have two `String`s, you need an `&`:

```rust
let hello = "Hello ".to_string();
let world = "world!".to_string();

let hello_world = hello + &world;
```

This is because `&String` can automatically coerce to a `&str`. This is a
feature called ‘[`Deref` coercions][dc]’.

[dc]: deref-coercions.html
[connect]: ../std/net/struct.TcpStream.html#method.connect
