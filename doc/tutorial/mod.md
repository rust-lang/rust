# Modules and crates

The Rust namespace is divided into modules. Each source file starts
with its own, empty module.

## Local modules

The `mod` keyword can be used to open a new, local module. In the
example below, `chicken` lives in the module `farm`, so, unless you
explicitly import it, you must refer to it by its long name,
`farm::chicken`.

    mod farm {
        fn chicken() -> str { "cluck cluck" }
        fn cow() -> str { "mooo" }
    }
    fn main() {
        log_err farm::chicken();
    }

Modules can be nested to arbitrary depth.

## Crates

The unit of independent compilation in Rust is the crate. Libraries
tend to be packaged as crates, and your own programs may consist of
one or more crates.

When compiling a single `.rs` file, the file acts as the whole crate.
You can compile it with the `--lib` compiler switch to create a shared
library, or without, provided that your file contains a `fn main`
somewhere, to create an executable.

It is also possible to include multiple files in a crate. For this
purpose, you create a `.rc` crate file, which references any number of
`.rs` code files. A crate file could look like this:

    #[link(name = "farm", vers = "2.5", author = "mjh")];
    mod cow;
    mod chicken;
    mod horse;

Compiling this file will cause `rustc` to look for files named
`cow.rs`, `chicken.rs`, `horse.rs` in the same directory as the `.rc`
file, compile them all together, and, depending on the presence of the
`--lib` switch, output a shared library or an executable.

The `#[link(...)]` part provides meta information about the module,
which other crates can use to load the right module. More about that
in a moment.

To have a nested directory structure for your source files, you can
nest mods in your `.rc` file:

    mod poultry {
        mod chicken;
        mod turkey;
    }

The compiler will now look for `poultry/chicken.rs` and
`poultry/turkey.rs`, and export their content in `poultry::chicken`
and `poultry::turkey`. You can also provide a `poultry.rs` to add
content to the `poultry` module itself.

## Using other crates

Having compiled a crate with `--lib`, you can use it in another crate
with a `use` directive. We've already seen `use std` in several of the
examples, which loads in the standard library.

`use` directives can appear in a crate file, or at the top level of a
single-file `.rs` crate. They will cause the compiler to search its
library search path (which you can extend with `-L` switch) for a Rust
crate library with the right name. This name is deduced from the crate
name in a platform-dependent way. The `farm` library will be called
`farm.dll` on Windows, `libfarm.so` on Linux, and `libfarm.dylib` on
OS X.

It is possible to provide more specific information when using an
external crate.

    use myfarm (name = "farm", vers = "2.7");

When a comma-separated list of name/value pairs is given after `use`,
these are matched against the attributes provided in the `link`
attribute of the crate file, and a crate is only used when the two
match. A `name` value can be given to override the name used to search
for the crate. So the above would import the `farm` crate under the
local name `myfarm`.

Our example crate declared this set of `link` attributes:

    #[link(name = "farm", vers = "2.5", author = "mjh")];

The version does not match the one provided in the `use` directive, so
unless the compiler can find another crate with the right version
somewhere, it will complain that no matching crate was found.

## A minimal example

Now for something that you can actually compile yourself. We have
these two files:

    // mylib.rs
    #[link(name = "mylib", vers = "1.0")];
    fn world() -> str { "world" }

    // main.rs
    use mylib;
    fn main() { log_err "hello " + mylib::world(); }

Now compile and run like this (adjust to your platform if necessary):

    > rustc --lib mylib.rs
    > rustc main.rs -L .
    > ./main
    "hello world"

## Importing

When using identifiers from other modules, it can get tiresome to
qualify them with the full module path every time (especially when
that path is several modules deep). Rust allows you to import
identifiers at the top of a file or module.

    use std;
    import std::io::println;
    fn main() {
        println("that was easy");
    }

It is also possible to import just the name of a module (`import
std::io;`, then use `io::println`), import all identifiers exported by
a given module (`import std::io::*`), or to import a specific set of
identifiers (`import std::math::{min, max, pi}`).

It is also possible to rename an identifier when importing, using the
`=` operator:

    import prnt = std::io::println;

## Exporting

By default, a module exports everything that it defines. This can be
restricted with `export` directives at the top of the module or file.

    mod enc {
        export encrypt, decrypt;
        const super_secret_number: int = 10;
        fn encrypt(n: int) { n + super_secret_number }
        fn decrypt(n: int) { n - super_secret_number }
    }

This defines a rock-solid encryption algorithm. Code outside of the
module can refer to the `enc::encrypt` and `enc::decrypt` identifiers
just fine, but it does not have access to `enc::super_secret_number`.

## Namespaces

Rust uses three different namespaces. One for modules, one for types,
and one for values. This means that this code is valid:

    mod buffalo {
        type buffalo = int;
        fn buffalo(buffalo: buffalo) -> buffalo { buffalo }
    }
    fn main() {
        let buffalo: buffalo::buffalo = 1;
        buffalo::buffalo(buffalo::buffalo(buffalo));
    }

You don't want to write things like that, but it *is* very practical
to not have to worry about name clashes between types, values, and
modules. This allows us to have a module `std::str`, for example, even
though `str` is a built-in type name.

## Resolution

The resolution process in Rust simply goes up the chain of contexts,
looking for the name in each context. Nested functions and modules
create new contexts inside their parent function or module. A file
that's part of a bigger crate will have that crate's context as parent
context.

Identifiers can shadow each others. In this program, `x` is of type
`int`:

    type t = str;
    fn main() {
        type t = int;
        let x: t;
    }

An `import` directive will only import into the namespaces for which
identifiers are actually found. Consider this example:

    type bar = uint;
    mod foo { fn bar() {} }
    mod baz {
        import foo::bar;
        const x: bar = 20u;
    }

When resolving the type name `bar` in the `const` definition, the
resolver will first look at the module context for `baz`. This has an
import named `bar`, but that's a function, not a type, So it continues
to the top level and finds a type named `bar` defined there.

Normally, multiple definitions of the same identifier in a scope are
disallowed. Local variables defined with `let` are an exception to
this—multiple `let` directives can redefine the same variable in a
single scope. When resolving the name of such a variable, the most
recent definition is used.

    fn main() {
        let x = 10;
        let x = x + 10;
        assert x == 20;
    }

This makes it possible to rebind a variable without actually mutating
it, which is mostly useful for destructuring (which can rebind, but
not assign).
