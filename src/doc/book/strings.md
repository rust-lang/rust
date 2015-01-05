% Strings

Strings are an important concept for any programmer to master. Rust's string
handling system is a bit different from other languages, due to its systems
focus. Any time you have a data structure of variable size, things can get
tricky, and strings are a re-sizable data structure. That being said, Rust's
strings also work differently than in some other systems languages, such as C.

Let's dig into the details. A **string** is a sequence of Unicode scalar values
encoded as a stream of UTF-8 bytes. All strings are guaranteed to be
validly encoded UTF-8 sequences. Additionally, strings are not null-terminated
and can contain null bytes.

Rust has two main types of strings: `&str` and `String`.

The first kind is a `&str`. This is pronounced a 'string slice.' String literals
are of the type `&str`:

```{rust}
let string = "Hello there."; // string: &str
```

This string is statically allocated, meaning that it's saved inside our
compiled program, and exists for the entire duration it runs. The `string`
binding is a reference to this statically allocated string. String slices
have a fixed size, and cannot be mutated.

A `String`, on the other hand, is an in-memory string.  This string is
growable, and is also guaranteed to be UTF-8.

```{rust}
let mut s = "Hello".to_string(); // mut s: String
println!("{}", s);

s.push_str(", world.");
println!("{}", s);
```

You can get a `&str` view into a `String` with the `as_slice()` method:

```{rust}
fn takes_slice(slice: &str) {
    println!("Got: {}", slice);
}

fn main() {
    let s = "Hello".to_string();
    takes_slice(s.as_slice());
}
```

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

Viewing a `String` as a `&str` is cheap, but converting the `&str` to a
`String` involves allocating memory. No reason to do that unless you have to!

That's the basics of strings in Rust! They're probably a bit more complicated
than you are used to, if you come from a scripting language, but when the
low-level details matter, they really matter. Just remember that `String`s
allocate memory and control their data, while `&str`s are a reference to
another string, and you'll be all set.
