% Hello, Cargo!

Cargo is Rust’s build system and package manager, and Rustaceans use Cargo to
manage their Rust projects. Cargo manages three things: building your code,
downloading the libraries your code depends on, and building those libraries.
We call libraries your code needs ‘dependencies’, since your code depends on
them.

The simplest Rust programs don’t have any dependencies, so right now, you'd
only use the first part of its functionality. As you write more complex Rust
programs, you’ll want to add dependencies, and if you start off using Cargo,
that will be a lot easier to do.

As the vast, vast majority of Rust projects use Cargo, we will assume that
you’re using it for the rest of the book. Cargo comes installed with Rust
itself, if you used the official installers. If you installed Rust through some
other means, you can check if you have Cargo installed by typing:

```bash
$ cargo --version
```

Into a terminal. If you see a version number, great! If you see an error like
‘`command not found`’, then you should look at the documentation for the system
in which you installed Rust, to determine if Cargo is separate.

## Converting to Cargo

Let’s convert the Hello World program to Cargo. To Cargo-fy a project, you need
to do three things: 

1. Put your source file in the right directory.
2. Get rid of the old executable (`main.exe` on Windows, `main` everywhere else)
   and make a new one.
3. Make a Cargo configuration file.

Let's get started!

### Creating a new Executable and Source Directory

First, go back to your terminal, move to your *hello_world* directory, and
enter the following commands:

```bash
$ mkdir src
$ mv main.rs src/main.rs
$ rm main  # or 'del main.exe' on Windows
```

Cargo expects your source files to live inside a *src* directory, so do that
first. This leaves the top level project directory (in this case,
*hello_world*) for READMEs, license information, and anything else not related
to your code. In this way, using Cargo helps you keep your projects nice and
tidy. There's a place for everything, and everything is in its place. 

Now, copy *main.rs* to the *src* directory, and delete the compiled file you
created with `rustc`. As usual, replace `main` with `main.exe` if you're on
Windows.

This example retains `main.rs` as the source filename because it's creating an
executable. If you wanted to make a library instead, you'd name the file
`lib.rs`. This convention is used by Cargo to successfully compile your
projects, but it can be overridden if you wish. 

### Creating a Configuration File

Next, create a new file inside your *hello_world* directory, and call it
`Cargo.toml`.

Make sure to capitalize the `C` in `Cargo.toml`, or Cargo won't know what to do
with the configuration file. 

This file is in the *[TOML]* (Tom's Obvious, Minimal Language) format. TOML is
similar to INI, but has some extra goodies. According to the TOML docs, TOML
“aims to be a minimal configuration file format that's easy to read”, and so we
chose it as the format Cargo uses.

[TOML]: https://github.com/toml-lang/toml

Inside this file, type the following information:

```toml
[package]

name = "hello_world"
version = "0.0.1"
authors = [ "Your name <you@example.com>" ]
```

The first line, `[package]`, indicates that the following statements are
configuring a package. As we add more information to this file, we’ll add other
sections, but for now, we just have the package configuration.

The other three lines set the three bits of configuration that Cargo needs to
know to compile your program: its name, what version it is, and who wrote it.

Once you've added this information to the *Cargo.toml* file, save it to finish
creating the configuration file.

## Building and Running a Cargo Project 

With your *Cargo.toml* file in place in your project's root directory, you
should be ready to build and run your Hello World program! To do so, enter the
following commands:

```bash
$ cargo build
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
$ ./target/debug/hello_world
Hello, world!
```

Bam! If all goes well, `Hello, world!` should print to the terminal once more. 

You just built a project with `cargo build` and ran it with
`./target/debug/hello_world`, but you can actually do both in one step with
`cargo run` as follows:

```bash
$ cargo run
     Running `target/debug/hello_world`
Hello, world!
```

Notice that this example didn’t re-build the project. Cargo figured out that
the hasn’t changed, and so it just ran the binary. If you'd modified your
program, Cargo would have built the file before running it, and you would have
seen something like this:

```bash
$ cargo run
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
     Running `target/debug/hello_world`
Hello, world!
```

Cargo checks to see if any of your project’s files have been modified, and only
rebuilds your project if they’ve changed since the last time you built it.

With simple projects, Cargo doesn't bring a whole lot over just using `rustc`,
but it will become useful in future. When your projects get more complex,
you'll need to do more things to get all of the parts to properly compile. With
Cargo, you can just run `cargo build`, and it should work the right way.

## Building for Release

When your project is finally ready for release, you can use `cargo build
--release` to compile your project with optimizations. These optimizations make
your Rust code run faster, but turning them on makes your program take longer
to compile. This is why there are two different profiles, one for development,
and one for building the final program you’ll give to a user.

Running this command also causes Cargo to create a new file called
*Cargo.lock*, which looks like this:

```toml
[root]
name = "hello_world"
version = "0.0.1"
```

Cargo uses the *Cargo.lock* file to keep track of dependencies in your
application. This is the Hello World project's *Cargo.lock* file. This project
doesn't have dependencies, so the file is a bit sparse. Realistically, you
won't ever need to touch this file yourself; just let Cargo handle it.

That’s it! If you've been following along, you should have successfully built
`hello_world` with Cargo. 

Even though the project is simple, it now uses much of the real tooling you’ll
use for the rest of your Rust career. In fact, you can expect to start
virtually all Rust projects with some variation on the following commands:

```bash
$ git clone someurl.com/foo
$ cd foo
$ cargo build
```

## Making A New Cargo Project the Easy Way

You don’t have to go through that previous process every time you want to start
a new project! Cargo can quickly make a bare-bones project directory that you
can start developing in right away.

To start a new project with Cargo, enter `cargo new` at the command line:

```bash
$ cargo new hello_world --bin
```

This command passes `--bin` because the goal is to get straight to making an
executable application, as opposed to a library. Executables are often called
*binaries* (as in `/usr/bin`, if you’re on a Unix system).

Cargo has generated two files and one directory for us: a `Cargo.toml` and a
*src* directory with a *main.rs* file inside. These should look familliar,
they’re exactly what we created by hand, above.

This output is all you need to get started. First, open `Cargo.toml`. It should
look something like this:

```toml
[package]

name = "hello_world"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
```

Cargo has populated *Cargo.toml* with reasonable defaults based on the arguments
you gave it and your `git` global configuration. You may notice that Cargo has
also initialized the `hello_world` directory as a `git` repository.

Here’s what should be in `src/main.rs`:

```rust
fn main() {
    println!("Hello, world!");
}
```

Cargo has generated a "Hello World!" for you, and you’re ready to start coding! 

> Note: If you want to look at Cargo in more detail, check out the official [Cargo
guide], which covers all of its features.

[Cargo guide]: http://doc.crates.io/guide.html

# Closing Thoughts

This chapter covered the basics that will serve you well through the rest of
this book, and the rest of your time with Rust. Now that you’ve got the tools
down, we'll cover more about the Rust language itself. 

You have two options: Dive into a project with ‘[Learn Rust][learnrust]’, or
start from the bottom and work your way up with ‘[Syntax and
Semantics][syntax]’. More experienced systems programmers will probably prefer
‘Learn Rust’, while those from dynamic backgrounds may enjoy either. Different
people learn differently! Choose whatever’s right for you.

[learnrust]: learn-rust.html
[syntax]: syntax-and-semantics.html
