% The Rust Crates and Modules Guide

When a project starts getting large, it's considered a good software
engineering practice to split it up into a bunch of smaller pieces, and then
fit them together. It's also important to have a well-defined interface, so
that some of your functionality is private, and some is public. To facilitate
these kinds of things, Rust has a module system.

# Basic terminology: Crates and Modules

Rust has two distinct terms that relate to the module system: "crate" and
"module." A crate is synonymous with a 'library' or 'package' in other
languages. Hence "Cargo" as the name of Rust's package management tool: you
ship your crates to others with Cargo. Crates can produce an executable or a
shared library, depending on the project.

Each crate has an implicit "root module" that contains the code for that crate.
You can then define a tree of sub-modules under that root module. Modules allow
you to partition your code within the crate itself.

As an example, let's make a "phrases" crate, which will give us various phrases
in different languages. To keep things simple, we'll stick to "greetings" and
"farewells" as two kinds of phrases, and use English and Japanese (日本語） as
two languages for those phrases to be in. We'll use this module layout:

```text
                                +-----------+
                            +---| greetings |
                            |   +-----------+
              +---------+   |
              | english |---+
              +---------+   |   +-----------+
              |             +---| farewells |
+---------+   |                 +-----------+
| phrases |---+ 
+---------+   |                  +-----------+
              |              +---| greetings |
              +----------+   |   +-----------+
              | japanese |---+
              +----------+   |
                             |   +-----------+
                             +---| farewells |
                                 +-----------+
```

In this example, `phrases` is the name of our crate. All of the rest are
modules.  You can see that they form a tree, branching out from the crate
"root", which is the root of the tree: `phrases` itself.

Now that we have a plan, let's define these modules in code. To start,
generate a new crate with Cargo:

```bash
$ cargo new phrases
$ cd phrases
```

If you remember, this generates a simple project for us:

```bash
$ tree .
.
├── Cargo.toml
└── src
    └── lib.rs

1 directory, 2 files
```

`src/lib.rs` is our crate root, corresponding to the `phrases` in our diagram
above.

# Defining Modules

To define each of our modules, we use the `mod` keyword. Let's make our
`src/lib.rs` look like this:

```
// in src/lib.rs

mod english {
    mod greetings {

    }

    mod farewells {

    }
}

mod japanese {
    mod greetings {

    }

    mod farewells {

    }
}
```

After the `mod` keyword, you give the name of the module. Module names follow
the conventions for other Rust identifiers: `lower_snake_case`. The contents of
each module are within curly braces (`{}`).

Within a given `mod`, you can declare sub-`mod`s. We can refer to sub-modules
with double-colon (`::`) notation: our four nested modules are
`english::greetings`, `english::farewells`, `japanese::greetings`, and
`japanese::farewells`. Because these sub-modules are namespaced under their
parent module, the names don't conflict: `english::greetings` and
`japanese::greetings` are distinct, even though their names are both
`greetings`.

Because this crate does not have a `main()` function, and is called `lib.rs`,
Cargo will build this crate as a library:

```bash
$ cargo build
   Compiling phrases v0.0.1 (file:///home/you/projects/phrases)
$ ls target
deps  libphrases-a7448e02a0468eaa.rlib  native
```

`libphrase-hash.rlib` is the compiled crate. Before we see how to use this
crate from another crate, let's break it up into multiple files.

# Multiple file crates

If each crate were just one file, these files would get very large. It's often
easier to split up crates into multiple files, and Rust supports this in two
ways.

Instead of declaring a module like this:

```{rust,ignore}
mod english {
    // contents of our module go here
}
```

We can instead declare our module like this:

```{rust,ignore}
mod english;
```

If we do that, Rust will expect to find either a `english.rs` file, or a
`english/mod.rs` file with the contents of our module:

```{rust,ignore}
// contents of our module go here
```

Note that in these files, you don't need to re-declare the module: that's
already been done with the initial `mod` declaration.

Using these two techniques, we can break up our crate into two directories and
seven files:

```bash
$ tree .
.
├── Cargo.lock
├── Cargo.toml
├── src
│   ├── english
│   │   ├── farewells.rs
│   │   ├── greetings.rs
│   │   └── mod.rs
│   ├── japanese
│   │   ├── farewells.rs
│   │   ├── greetings.rs
│   │   └── mod.rs
│   └── lib.rs
└── target
    ├── deps
    ├── libphrases-a7448e02a0468eaa.rlib
    └── native
```

`src/lib.rs` is our crate root, and looks like this:

```{rust,ignore}
// in src/lib.rs

mod english;

mod japanese;
```

These two declarations tell Rust to look for either `src/english.rs` and
`src/japanese.rs`, or `src/english/mod.rs` and `src/japanese/mod.rs`, depending
on our preference. In this case, because our modules have sub-modules, we've
chosen the second. Both `src/english/mod.rs` and `src/japanese/mod.rs` look
like this:

```{rust,ignore}
// both src/english/mod.rs and src/japanese/mod.rs

mod greetings;

mod farewells;
```

Again, these declarations tell Rust to look for either
`src/english/greetings.rs` and `src/japanese/greetings.rs` or
`src/english/farewells/mod.rs` and `src/japanese/farewells/mod.rs`. Because
these sub-modules don't have their own sub-modules, we've chosen to make them
`src/english/greetings.rs` and `src/japanese/farewells.rs`. Whew!

Right now, the contents of `src/english/greetings.rs` and
`src/japanese/farewells.rs` are both empty at the moment. Let's add some
functions.

Put this in `src/english/greetings.rs`:

```rust
// in src/english/greetings.rs

fn hello() -> String {
    "Hello!".to_string()
}  
```

Put this in `src/english/farewells.rs`:

```rust
// in src/english/farewells.rs

fn goodbye() -> String {
    "Goodbye.".to_string()
} 
```

Put this in `src/japanese/greetings.rs`:

```rust
// in src/japanese/greetings.rs

fn hello() -> String {
    "こんにちは".to_string()
}  
```

Of course, you can copy and paste this from this web page, or just type
something else. It's not important that you actually put "konnichiwa" to learn
about the module system.

Put this in `src/japanese/farewells.rs`:

```rust
// in src/japanese/farewells.rs

fn goodbye() -> String {
    "さようなら".to_string()
} 
```

(This is "Sayoonara", if you're curious.)

Now that we have our some functionality in our crate, let's try to use it from
another crate.

# Importing External Crates

We have a library crate. Let's make an executable crate that imports and uses
our library.

Make a `src/main.rs` and put this in it: (it won't quite compile yet)

```rust,ignore
// in src/main.rs

extern crate phrases;

fn main() {
    println!("Hello in English: {}", phrases::english::greetings::hello());
    println!("Goodbye in English: {}", phrases::english::farewells::goodbye());

    println!("Hello in Japanese: {}", phrases::japanese::greetings::hello());
    println!("Goodbye in Japanese: {}", phrases::japanese::farewells::goodbye());
}
```

The `extern crate` declaration tells Rust that we need to compile and link to
the `phrases` crate. We can then use `phrases`' modules in this one. As we
mentioned earlier, you can use double colons to refer to sub-modules and the
functions inside of them.

Also, Cargo assumes that `src/main.rs` is the crate root of a binary crate,
rather than a library crate. Once we compile `src/main.rs`, we'll get an
executable that we can run. Our package now has two crates: `src/lib.rs` and
`src/main.rs`. This pattern is quite common for executable crates: most
functionality is in a library crate, and the executable crate uses that
library. This way, other programs can also use the library crate, and it's also
a nice separation of concerns.

This doesn't quite work yet, though. We get four errors that look similar to
this:

```bash
$ cargo build
   Compiling phrases v0.0.1 (file:///home/you/projects/phrases)
/home/you/projects/phrases/src/main.rs:4:38: 4:72 error: function `hello` is private
/home/you/projects/phrases/src/main.rs:4     println!("Hello in English: {}", phrases::english::greetings::hello());
                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
note: in expansion of format_args!
<std macros>:2:23: 2:77 note: expansion site
<std macros>:1:1: 3:2 note: in expansion of println!
/home/you/projects/phrases/src/main.rs:4:5: 4:76 note: expansion site

```

By default, everything is private in Rust. Let's talk about this in some more
depth.

# Exporting a Public Interface

Rust allows you to precisely control which aspects of your interface are
public, and so private is the default. To make things public, you use the `pub`
keyword. Let's focus on the `english` module first, so let's reduce our `src/main.rs`
to just this:

```{rust,ignore}
// in src/main.rs

extern crate phrases;

fn main() {
    println!("Hello in English: {}", phrases::english::greetings::hello());
    println!("Goodbye in English: {}", phrases::english::farewells::goodbye());
}
```

In our `src/lib.rs`, let's add `pub` to the `english` module declaration:

```{rust,ignore}
// in src/lib.rs

pub mod english;

mod japanese;
```

And in our `src/english/mod.rs`, let's make both `pub`:

```{rust,ignore}
// in src/english/mod.rs

pub mod greetings;

pub mod farewells;
```

In our `src/english/greetings.rs`, let's add `pub` to our `fn` declaration:

```{rust,ignore}
// in src/english/greetings.rs

pub fn hello() -> String {
    "Hello!".to_string()
}
```

And also in `src/english/farewells.rs`:

```{rust,ignore}
// in src/english/farewells.rs

pub fn goodbye() -> String {
    "Goodbye.".to_string()
}
```

Now, our crate compiles, albeit with warnings about not using the `japanese`
functions:

```bash
$ cargo run
   Compiling phrases v0.0.1 (file:///home/you/projects/phrases)
/home/you/projects/phrases/src/japanese/greetings.rs:1:1: 3:2 warning: code is never used: `hello`, #[warn(dead_code)] on by default
/home/you/projects/phrases/src/japanese/greetings.rs:1 fn hello() -> String {
/home/you/projects/phrases/src/japanese/greetings.rs:2     "こんにちは".to_string()
/home/you/projects/phrases/src/japanese/greetings.rs:3 } 
/home/you/projects/phrases/src/japanese/farewells.rs:1:1: 3:2 warning: code is never used: `goodbye`, #[warn(dead_code)] on by default
/home/you/projects/phrases/src/japanese/farewells.rs:1 fn goodbye() -> String {
/home/you/projects/phrases/src/japanese/farewells.rs:2     "さようなら".to_string()
/home/you/projects/phrases/src/japanese/farewells.rs:3 } 
     Running `target/phrases`
Hello in English: Hello!
Goodbye in English: Goodbye.
```

Now that our functions are public, we can use them. Great! However, typing out
`phrases::english::greetings::hello()` is very long and repetitive. Rust has
another keyword for importing names into the current scope, so that you can
refer to them with shorter names. Let's talk about `use`.

# Importing Modules with `use`

Rust has a `use` keyword, which allows us to import names into our local scope.
Let's change our `src/main.rs` to look like this:

```{rust,ignore}
// in src/main.rs

extern crate phrases;

use phrases::english::greetings;
use phrases::english::farewells;

fn main() {
    println!("Hello in English: {}", greetings::hello());
    println!("Goodbye in English: {}", farewells::goodbye());
}
```

The two `use` lines import each module into the local scope, so we can refer to
the functions by a much shorter name. By convention, when importing functions, it's
considered best practice to import the module, rather than the function directly. In
other words, you _can_ do this:

```{rust,ignore}
extern crate phrases;

use phrases::english::greetings::hello;
use phrases::english::farewells::goodbye;

fn main() {
    println!("Hello in English: {}", hello());
    println!("Goodbye in English: {}", goodbye());
}
```

But it is not idiomatic. This is significantly more likely to introducing a
naming conflict. In our short program, it's not a big deal, but as it grows, it
becomes a problem. If we have conflicting names, Rust will give a compilation
error. For example, if we made the `japanese` functions public, and tried to do
this:

```{rust,ignore}
extern crate phrases;

use phrases::english::greetings::hello;
use phrases::japanese::greetings::hello;

fn main() {
    println!("Hello in English: {}", hello());
    println!("Hello in Japanese: {}", hello());
}
```

Rust will give us a compile-time error:

```{rust,ignore}
   Compiling phrases v0.0.1 (file:///home/you/projects/phrases)
/home/you/projects/phrases/src/main.rs:4:5: 4:40 error: a value named `hello` has already been imported in this module
/home/you/projects/phrases/src/main.rs:4 use phrases::japanese::greetings::hello;
                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
error: aborting due to previous error
Could not compile `phrases`.
```

If we're importing multiple names from the same module, we don't have to type it out
twice. Rust has a shortcut syntax for writing this:

```{rust,ignore}
use phrases::english::greetings;
use phrases::english::farewells;
```

You use curly braces:

```{rust,ignore}
use phrases::english::{greetings, farewells};
```

These two declarations are equivalent, but the second is a lot less typing.

## Re-exporting with `pub use`

You don't just use `use` to shorten identifiers. You can also use it inside of your crate
to re-export a function inside another module. This allows you to present an external
interface that may not directly map to your internal code organization.

Let's look at an example. Modify your `src/main.rs` to read like this:

```{rust,ignore}
// in src/main.rs

extern crate phrases;

use phrases::english::{greetings,farewells};
use phrases::japanese;

fn main() {
    println!("Hello in English: {}", greetings::hello());
    println!("Goodbye in English: {}", farewells::goodbye());

    println!("Hello in Japanese: {}", japanese::hello());
    println!("Goodbye in Japanese: {}", japanese::goodbye());
}
```

Then, modify your `src/lib.rs` to make the `japanese` mod public:

```{rust,ignore}
// in src/lib.rs

pub mod english;

pub mod japanese;
```

Next, make the two functions public, first in `src/japanese/greetings.rs`:

```{rust,ignore}
// in src/japanese/greetings.rs

pub fn hello() -> String {
    "こんにちは".to_string()
}
```

And then in `src/japanese/farewells.rs`:

```{rust,ignore}
// in src/japanese/farewells.rs

pub fn goodbye() -> String {
    "さようなら".to_string()
}
```

Finally, modify your `src/japanese/mod.rs` to read like this:

```{rust,ignore}
// in src/japanese/mod.rs

pub use self::greetings::hello;
pub use self::farewells::goodbye;

mod greetings;

mod farewells;
```

The `pub use` declaration brings the function into scope at this part of our
module hierarchy. Because we've `pub use`d this inside of our `japanese`
module, we now have a `phrases::japanese::hello()` function and a
`phrases::japanese::goodbye()` function, even though the code for them lives in
`phrases::japanese::greetings::hello()` and
`phrases::japanese::farewells::goodbye()`. Our internal organization doesn't
define our external interface.

Also, note that we `pub use`d before we declared our `mod`s. Rust requires that
`use` declarations go first.

This will build and run:

```bash
$ cargo build
   Compiling phrases v0.0.1 (file:///home/you/projects/phrases)
     Running `target/phrases`
Hello in English: Hello!
Goodbye in English: Goodbye.
Hello in Japanese: こんにちは
Goodbye in Japanese: さようなら
```
