% Hello, Cargo!

[Cargo][cratesio] is a tool that Rustaceans use to help manage their Rust
projects. Cargo is currently in a pre-1.0 state, and so it is still a work in
progress. However, it is already good enough to use for many Rust projects, and
so it is assumed that Rust projects will use Cargo from the beginning.

[cratesio]: https://doc.crates.io

Cargo manages three things: building your code, downloading the dependencies
your code needs, and building those dependencies. At first, your
program doesn’t have any dependencies, so we’ll only be using the first part of
its functionality. Eventually, we’ll add more. Since we started off by using
Cargo, it'll be easy to add later.

If you installed Rust via the official installers you will also have Cargo. If
you installed Rust some other way, you may want to [check the Cargo
README][cargoreadme] for specific instructions about installing it.

[cargoreadme]: https://github.com/rust-lang/cargo#installing-cargo-from-nightlies

## Converting to Cargo

Let’s convert Hello World to Cargo.

To Cargo-ify our project, we need to do two things: Make a `Cargo.toml`
configuration file, and put our source file in the right place. Let's
do that part first:

```bash
$ mkdir src
$ mv main.rs src/main.rs
```

Cargo expects your source files to live inside a `src` directory. That leaves
the top level for other things, like READMEs, license information, and anything
not related to your code. Cargo helps us keep our projects nice and tidy. A
place for everything, and everything in its place.

Next, our configuration file:

```bash
$ editor Cargo.toml
```

Make sure to get this name right: you need the capital `C`!

Put this inside:

```toml
[package]

name = "hello_world"
version = "0.0.1"
authors = [ "Your name <you@example.com>" ]
```

This file is in the [TOML][toml] format. Let’s let it explain itself to you:

> TOML aims to be a minimal configuration file format that's easy to read due
> to obvious semantics. TOML is designed to map unambiguously to a hash table.
> TOML should be easy to parse into data structures in a wide variety of
> languages.

TOML is very similar to INI, but with some extra goodies.

[toml]: https://github.com/toml-lang/toml

Once you have this file in place, we should be ready to build! Try this:

```bash
$ cargo build
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
$ ./target/hello_world
Hello, world!
```

Bam! We build our project with `cargo build`, and run it with
`./target/hello_world`. We can do both in one step with `cargo run`:

```bash
$ cargo run
     Running `target/hello_world`
Hello, world!
```

Notice that we didn’t re-build the project this time. Cargo figured out that
we hadn’t changed the source file, and so it just ran the binary. If we had
made a modification, we would have seen it do both:

```bash
$ cargo build
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
     Running `target/hello_world`
Hello, world!
```

This hasn’t bought us a whole lot over our simple use of `rustc`, but think
about the future: when our project gets more complex, we would need to do more
things to get all of the parts to properly compile. With Cargo, as our project
grows, we can just `cargo build`, and it’ll work the right way.

When your project is finally ready for release, you can use
`cargo build --release` to compile your project with optimizations.

You'll also notice that Cargo has created a new file: `Cargo.lock`.

```toml
[root]
name = "hello_world"
version = "0.0.1"
```

This file is used by Cargo to keep track of dependencies in your application.
Right now, we don’t have any, so it’s a bit sparse. You won't ever need
to touch this file yourself, just let Cargo handle it.

That’s it! We’ve successfully built `hello_world` with Cargo. Even though our
program is simple, it’s using much of the real tooling that you’ll use for the
rest of your Rust career. You can expect to do this to get started with
virtually all Rust projects:

```bash
$ git clone someurl.com/foo
$ cd foo
$ cargo build
```

## A New Project

You don’t have to go through this whole process every time you want to start a
new project! Cargo has the ability to make a bare-bones project directory in
which you can start developing right away.

To start a new project with Cargo, use `cargo new`:

```bash
$ cargo new hello_world --bin
```

We’re passing `--bin` because we're making a binary program: if we were making
a library, we'd leave it off.

Let's check out what Cargo has generated for us:

```bash
$ cd hello_world
$ tree .
.
├── Cargo.toml
└── src
    └── main.rs

1 directory, 2 files
```

If you don't have the `tree` command, you can probably get it from your
distribution’s package manager. It’s not necessary, but it’s certainly useful.

This is all we need to get started. First, let’s check out `Cargo.toml`:

```toml
[package]

name = "hello_world"
version = "0.0.1"
authors = ["Your Name <you@example.com>"]
```

Cargo has populated this file with reasonable defaults based off the arguments
you gave it and your `git` global configuration. You may notice that Cargo has
also initialized the `hello_world` directory as a `git` repository.

Here’s what’s in `src/main.rs`:

```rust
fn main() {
    println!("Hello, world!");
}
```

Cargo has generated a "Hello World!" for us, and you’re ready to start coding! Cargo
has its own [guide][guide] which covers Cargo’s features in much more depth.

[guide]: http://doc.crates.io/guide.html

Now that you’ve got the tools down, let’s actually learn more about the Rust
language itself. These are the basics that will serve you well through the rest
of your time with Rust.

You have two options: Dive into a project with ‘[Learn Rust][learnrust]’, or
start from the bottom and work your way up with ‘[Syntax and
Semantics][syntax]’. More experienced systems programmers will probably prefer
‘Learn Rust’, while those from dynamic backgrounds may enjoy either. Different
people learn differently! Choose whatever’s right for you.

[learnrust]: learn-rust.html
[syntax]: syntax-and-semantics.html
