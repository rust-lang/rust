% Rust Cheatsheet

# How do I convert *X* to *Y*?

**Int to string**

Use [`ToStr`](http://static.rust-lang.org/doc/master/std/to_str/trait.ToStr.html).

~~~
let x: int = 42;
let y: ~str = x.to_str();
~~~

**String to int**

Use [`FromStr`](http://static.rust-lang.org/doc/master/std/from_str/trait.FromStr.html), and its helper function, [`from_str`](http://static.rust-lang.org/doc/master/std/from_str/fn.from_str.html).

~~~
let x: Option<int> = from_str("42");
let y: int = x.unwrap();
~~~

**Int to string, in non-base-10**

Use [`ToStrRadix`](http://static.rust-lang.org/doc/master/std/num/trait.ToStrRadix.html).

~~~
use std::num::ToStrRadix;

let x: int = 42;
let y: ~str = x.to_str_radix(16);
~~~

**String to int, in non-base-10**

Use [`FromStrRadix`](http://static.rust-lang.org/doc/master/std/num/trait.FromStrRadix.html), and its helper function, [`from_str_radix`](http://static.rust-lang.org/doc/master/std/num/fn.from_str_radix.html).

~~~
use std::num;

let x: Option<i64> = num::from_str_radix("deadbeef", 16);
let y: i64 = x.unwrap();
~~~

**Vector of Bytes to String**

To return a Borrowed String Slice (&str) use the str helper function [`from_utf8`](http://static.rust-lang.org/doc/master/std/str/fn.from_utf8.html).

~~~
use std::str;

let bytes = ~[104u8,105u8];
let x: Option<&str> = str::from_utf8(bytes);
let y: &str = x.unwrap();
~~~

To return an Owned String (~str) use the str helper function [`from_utf8_owned`](http://static.rust-lang.org/doc/master/std/str/fn.from_utf8_owned.html).

~~~
use std::str;

let x: Option<~str> = str::from_utf8_owned(~[104u8,105u8]);
let y: ~str = x.unwrap();
~~~~

To return a [`MaybeOwned`](http://static.rust-lang.org/doc/master/std/str/enum.MaybeOwned.html) use the str helper function [`from_utf8_lossy`](http://static.rust-lang.org/doc/master/std/str/fn.from_utf8_owned.html).  This function also replaces non-valid utf-8 sequences with U+FFFD replacement character.

~~~
use std::str;

let x = bytes!(72u8,"ello ",0xF0,0x90,0x80,"World!");
let y = str::from_utf8_lossy(x);
~~~~

# File operations

## How do I read from a file?

Use [`File::open`](http://static.rust-lang.org/doc/master/std/io/fs/struct.File.html#method.open) to create a [`File`](http://static.rust-lang.org/doc/master/std/io/fs/struct.File.html) struct, which implements the [`Reader`](http://static.rust-lang.org/doc/master/std/io/trait.Reader.html) trait.

~~~ {.ignore}
use std::path::Path;
use std::io::fs::File;

let path : Path   = Path::new("Doc-FAQ-Cheatsheet.md");
let on_error      = || fail!("open of {:?} failed", path);
let reader : File = File::open(&path).unwrap_or_else(on_error);
~~~

## How do I iterate over the lines in a file?

Use the [`lines`](http://static.rust-lang.org/doc/master/std/io/trait.Buffer.html#method.lines) method on a [`BufferedReader`](http://static.rust-lang.org/doc/master/std/io/buffered/struct.BufferedReader.html).

~~~
use std::io::BufferedReader;
# use std::io::MemReader;

# let reader = MemReader::new(~[]);

let mut reader = BufferedReader::new(reader);
for line in reader.lines() {
    print!("line: {}", line);
}
~~~

# String operations

## How do I search for a substring?

Use the [`find_str`](http://static.rust-lang.org/doc/master/std/str/trait.StrSlice.html#tymethod.find_str) method.

~~~
let str = "Hello, this is some random string";
let index: Option<uint> = str.find_str("rand");
~~~

# Containers

## How do I get the length of a vector?

The [`Container`](http://static.rust-lang.org/doc/master/std/container/trait.Container.html) trait provides the `len` method.

~~~
let u: ~[u32] = ~[0, 1, 2];
let v: &[u32] = &[0, 1, 2, 3];
let w: [u32, .. 5] = [0, 1, 2, 3, 4];

println!("u: {}, v: {}, w: {}", u.len(), v.len(), w.len()); // 3, 4, 5
~~~

## How do I iterate over a vector?

Use the [`iter`](http://static.rust-lang.org/doc/master/std/vec/trait.ImmutableVector.html#tymethod.iter) method.

~~~
let values: ~[int] = ~[1, 2, 3, 4, 5];
for value in values.iter() {  // value: &int
    println!("{}", *value);
}
~~~

(See also [`mut_iter`](http://static.rust-lang.org/doc/master/std/vec/trait.MutableVector.html#tymethod.mut_iter) which yields `&mut int` and [`move_iter`](http://static.rust-lang.org/doc/master/std/vec/trait.OwnedVector.html#tymethod.move_iter) which yields `int` while consuming the `values` vector.)

# Type system

## How do I store a function in a struct?

~~~
struct Foo {
    myfunc: fn(int, uint) -> i32
}

struct FooClosure<'a> {
    myfunc: 'a |int, uint| -> i32
}

fn a(a: int, b: uint) -> i32 {
    (a as uint + b) as i32
}

fn main() {
    let f = Foo { myfunc: a };
    let g = FooClosure { myfunc: |a, b|  { (a - b as int) as i32 } };
    println!("{}", (f.myfunc)(1, 2));
    println!("{}", (g.myfunc)(3, 4));
}
~~~

Note that the parenthesis surrounding `f.myfunc` are necessary: they are how Rust disambiguates field lookup and method call. The `'a` on `FooClosure` is the lifetime of the closure's environment pointer.

## How do I express phantom types?

[Phantom types](http://www.haskell.org/haskellwiki/Phantom_type) are those that cannot be constructed at compile time. To express these in Rust, zero-variant `enum`s can be used:

~~~
enum Open {}
enum Closed {}
~~~

Phantom types are useful for enforcing state at compile time. For example:

~~~
struct Door<State>(~str);

struct Open;
struct Closed;

fn close(Door(name): Door<Open>) -> Door<Closed> {
    Door::<Closed>(name)
}

fn open(Door(name): Door<Closed>) -> Door<Open> {
    Door::<Open>(name)
}

let _ = close(Door::<Open>(~"front"));
~~~

Attempting to close a closed door is prevented statically:

~~~ {.ignore}
let _ = close(Door::<Closed>(~"front")); // error: mismatched types: expected `main::Door<main::Open>` but found `main::Door<main::Closed>`
~~~

# FFI (Foreign Function Interface)

## C function signature conversions

Description            C signature                                    Equivalent Rust signature
---------------------- ---------------------------------------------- ------------------------------------------
no parameters          `void foo(void);`                              `fn foo();`
return value           `int foo(void);`                               `fn foo() -> c_int;`
function parameters    `void foo(int x, int y);`                      `fn foo(x: int, y: int);`
in-out pointers        `void foo(const int* in_ptr, int* out_ptr);`   `fn foo(in_ptr: *c_int, out_ptr: *mut c_int);`

Note: The Rust signatures should be wrapped in an `extern "ABI" { ... }` block.

### Representing opaque handles

You might see things like this in C APIs:

~~~ {.notrust}
typedef struct Window Window;
Window* createWindow(int width, int height);
~~~

You can use a zero-element `enum` ([phantom type](#how-do-i-express-phantom-types)) to represent the opaque object handle. The FFI would look like this:

~~~ {.ignore}
enum Window {}
extern "C" {
    fn createWindow(width: c_int, height: c_int) -> *Window;
}
~~~

Using a phantom type ensures that the handles cannot be (safely) constructed in client code.

# Contributing to this page

For small examples, have full type annotations, as much as is reasonable, to keep it clear what, exactly, everything is doing. Try to link to the API docs, as well.

Similar documents for other programming languages:

  * [http://pleac.sourceforge.net/](http://pleac.sourceforge.net)
