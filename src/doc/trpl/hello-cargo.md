% Hello, Cargo!

[Cargo](http://crates.io) is a tool that Rustaceans use to help manage their
Rust projects. Cargo is currently in an alpha state, just like Rust, and so it
is still a work in progress. However, it is already good enough to use for many
Rust projects, and so it is assumed that Rust projects will use Cargo from the
beginning.

Cargo manages three things: building your code, downloading the dependencies
your code needs, and building the dependencies your code needs.  At first, your
program doesn't have any dependencies, so we'll only be using the first part of
its functionality. Eventually, we'll add more. Since we started off by using
Cargo, it'll be easy to add later.

If you installed Rust via the official installers you will also have
Cargo. If you installed Rust some other way, you may want to [check
the Cargo
README](https://github.com/rust-lang/cargo#installing-cargo-from-nightlies)
for specific instructions about installing it.

Let's convert Hello World to Cargo.

To Cargo-ify our project, we need to do two things: Make a `Cargo.toml`
configuration file, and put our source file in the right place. Let's
do that part first:

```{bash}
$ mkdir src
$ mv main.rs src/main.rs
```

Cargo expects your source files to live inside a `src` directory. That leaves
the top level for other things, like READMEs, license information, and anything
not related to your code. Cargo helps us keep our projects nice and tidy. A
place for everything, and everything in its place.

Next, our configuration file:

```{bash}
$ editor Cargo.toml
```

Make sure to get this name right: you need the capital `C`!

Put this inside:

```toml
[package]

name = "hello_world"
version = "0.0.1"
authors = [ "Your name <you@example.com>" ]

[[bin]]

name = "hello_world"
```

This file is in the [TOML](https://github.com/toml-lang/toml) format. Let's let
it explain itself to you:

> TOML aims to be a minimal configuration file format that's easy to read due
> to obvious semantics. TOML is designed to map unambiguously to a hash table.
> TOML should be easy to parse into data structures in a wide variety of
> languages.

TOML is very similar to INI, but with some extra goodies.

Anyway, there are two **table**s in this file: `package` and `bin`. The first
tells Cargo metadata about your package. The second tells Cargo that we're
interested in building a binary, not a library (though we could do both!), as
well as what it is named.

Once you have this file in place, we should be ready to build! Try this:

```{bash}
$ cargo build
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
$ ./target/hello_world
Hello, world!
```

Bam! We build our project with `cargo build`, and run it with
`./target/hello_world`. This hasn't bought us a whole lot over our simple use
of `rustc`, but think about the future: when our project has more than one
file, we would need to call `rustc` more than once, and pass it a bunch of options to
tell it to build everything together. With Cargo, as our project grows, we can
just `cargo build` and it'll work the right way.

You'll also notice that Cargo has created a new file: `Cargo.lock`.

```toml
[root]
name = "hello_world"
version = "0.0.1"
```

This file is used by Cargo to keep track of dependencies in your application.
Right now, we don't have any, so it's a bit sparse. You won't ever need
to touch this file yourself, just let Cargo handle it.

That's it! We've successfully built `hello_world` with Cargo. Even though our
program is simple, it's using much of the real tooling that you'll use for the
rest of your Rust career.

Now that you've got the tools down, let's actually learn more about the Rust
language itself. These are the basics that will serve you well through the rest
of your time with Rust.
