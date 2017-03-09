# Getting Started

This first chapter of the book will get us going with Rust and its tooling.
First, we’ll install Rust. Then, the classic ‘Hello World’ program. Finally,
we’ll talk about Cargo, Rust’s build system and package manager.

We’ll be showing off a number of commands using a terminal, and those lines all
start with `$`. You don't need to type in the `$`s, they are there to indicate
the start of each command. We’ll see many tutorials and examples around the web
that follow this convention: `$` for commands run as our regular user, and `#`
for commands we should be running as an administrator.

# Installing Rust

The first step to using Rust is to install it. Generally speaking, you’ll need
an Internet connection to run the commands in this section, as we’ll be
downloading Rust from the Internet.

The Rust compiler runs on, and compiles to, a great number of platforms, but is
best supported on Linux, Mac, and Windows, on the x86 and x86-64 CPU
architecture. There are official builds of the Rust compiler and standard
library for these platforms and more. [For full details on Rust platform support
see the website][platform-support].

[platform-support]: https://forge.rust-lang.org/platform-support.html

## Installing Rust

All you need to do on Unix systems like Linux and macOS is open a
terminal and type this:

```bash
$ curl https://sh.rustup.rs -sSf | sh
```

It will download a script, and start the installation. If everything
goes well, you’ll see this appear:

```text
Rust is installed now. Great! 
```

Installing on Windows is nearly as easy: download and run
[rustup-init.exe]. It will start the installation in a console and
present the above message on success.

For other installation options and information, visit the [install]
page of the Rust website.

[rustup-init.exe]: https://win.rustup.rs
[install]: https://www.rust-lang.org/install.html

## Uninstalling

Uninstalling Rust is as easy as installing it:

```bash
$ rustup self uninstall
```

## Troubleshooting

If we've got Rust installed, we can open up a shell, and type this:

```bash
$ rustc --version
```

You should see the version number, commit hash, and commit date.

If you do, Rust has been installed successfully! Congrats!

If you don't, that probably means that the `PATH` environment variable
doesn't include Cargo's binary directory, `~/.cargo/bin` on Unix, or
`%USERPROFILE%\.cargo\bin` on Windows. This is the directory where
Rust development tools live, and most Rust developers keep it in their
`PATH` environment variable, which makes it possible to run `rustc` on
the command line. Due to differences in operating systems, command
shells, and bugs in installation, you may need to restart your shell,
log out of the system, or configure `PATH` manually as appropriate for
your operating environment.

Rust does not do its own linking, and so you’ll need to have a linker
installed. Doing so will depend on your specific system. For
Linux-based systems, Rust will attempt to call `cc` for linking. On
`windows-msvc` (Rust built on Windows with Microsoft Visual Studio),
this depends on having [Microsoft Visual C++ Build Tools][msvbt]
installed. These do not need to be in `%PATH%` as `rustc` will find
them automatically. In general, if you have your linker in a
non-traditional location you can call `rustc 
linker=/path/to/cc`, where `/path/to/cc` should point to your linker path.

[msvbt]: http://landinghub.visualstudio.com/visual-cpp-build-tools

If you are still stuck, there are a number of places where we can get
help. The easiest is
[the #rust-beginners IRC channel on irc.mozilla.org][irc-beginners] 
and for general discussion
[the #rust IRC channel on irc.mozilla.org][irc], which we 
can access through [Mibbit][mibbit]. Then we'll be chatting with other
Rustaceans (a silly nickname we call ourselves) who can help us out. Other great
resources include [the user’s forum][users] and [Stack Overflow][stackoverflow].

[irc-beginners]: irc://irc.mozilla.org/#rust-beginners
[irc]: irc://irc.mozilla.org/#rust
[mibbit]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-beginners,%23rust
[users]: https://users.rust-lang.org/
[stackoverflow]: http://stackoverflow.com/questions/tagged/rust

This installer also installs a copy of the documentation locally, so we can
read it offline. It's only a `rustup doc` away!

# Hello, world!

Now that you have Rust installed, we'll help you write your first Rust program.
It's traditional when learning a new language to write a little program to
print the text “Hello, world!” to the screen, and in this section, we'll follow
that tradition.

The nice thing about starting with such a simple program is that you can
quickly verify that your compiler is installed, and that it's working properly.
Printing information to the screen is also a pretty common thing to do, so
practicing it early on is good.

> Note: This book assumes basic familiarity with the command line. Rust itself
> makes no specific demands about your editing, tooling, or where your code
> lives, so if you prefer an IDE to the command line, that's an option. You may
> want to check out [SolidOak], which was built specifically with Rust in mind.
> There are a number of extensions in development by the community, and the
> Rust team ships plugins for [various editors]. Configuring your editor or
> IDE is out of the scope of this tutorial, so check the documentation for your
> specific setup.

[SolidOak]: https://github.com/oakes/SolidOak
[various editors]: https://github.com/rust-lang/rust/blob/master/src/etc/CONFIGS.md

## Creating a Project File

First, make a file to put your Rust code in. Rust doesn't care where your code
lives, but for this book, I suggest making a *projects* directory in your home
directory, and keeping all your projects there. Open a terminal and enter the
following commands to make a directory for this particular project:

```bash
$ mkdir ~/projects
$ cd ~/projects
$ mkdir hello_world
$ cd hello_world
```

> Note: If you’re on Windows and not using PowerShell, the `~` may not work.
> Consult the documentation for your shell for more details.

## Writing and Running a Rust Program

We need to create a source file for our Rust program. Rust files always end
in a *.rs* extension. If you are using more than one word in your filename,
use an underscore to separate them; for example, you would use
*my_program.rs* rather than *myprogram.rs*.

Now, make a new file and call it *main.rs*. Open the file and type
the following code:

```rust
fn main() {
    println!("Hello, world!");
}
```

Save the file, and go back to your terminal window. On Linux or macOS, enter the
following commands:

```bash
$ rustc main.rs
$ ./main
Hello, world!
```

In Windows, replace `main` with `main.exe`. Regardless of your operating
system, you should see the string `Hello, world!` print to the terminal. If you
did, then congratulations! You've officially written a Rust program. That makes
you a Rust programmer! Welcome.

## Anatomy of a Rust Program

Now, let’s go over what just happened in your "Hello, world!" program in
detail. Here's the first piece of the puzzle:

```rust
fn main() {

}
```

These lines define a *function* in Rust. The `main` function is special: it's
the beginning of every Rust program. The first line says, “I’m declaring a
function named `main` that takes no arguments and returns nothing.” If there
were arguments, they would go inside the parentheses (`(` and `)`), and because
we aren’t returning anything from this function, we can omit the return type
entirely.

Also note that the function body is wrapped in curly braces (`{` and `}`). Rust
requires these around all function bodies. It's considered good style to put
the opening curly brace on the same line as the function declaration, with one
space in between.

Inside the `main()` function:

```rust
    println!("Hello, world!");
```

This line does all of the work in this little program: it prints text to the
screen. There are a number of details that are important here. The first is
that it’s indented with four spaces, not tabs.

The second important part is the `println!()` line. This is calling a Rust
*[macro]*, which is how metaprogramming is done in Rust. If it were calling a
function instead, it would look like this: `println()` (without the !). We'll
discuss Rust macros in more detail later, but for now you only need to
know that when you see a `!` that means that you’re calling a macro instead of
a normal function.


[macro]: macros.html

Next is `"Hello, world!"` which is a *string*. Strings are a surprisingly
complicated topic in a systems programming language, and this is a *[statically
allocated]* string. We pass this string as an argument to `println!`, which
prints the string to the screen. Easy enough!

[statically allocated]: the-stack-and-the-heap.html

The line ends with a semicolon (`;`). Rust is an *[expression-oriented
language]*, which means that most things are expressions, rather than
statements. The `;` indicates that this expression is over, and the next one is
ready to begin. Most lines of Rust code end with a `;`.

[expression-oriented language]: glossary.html#expression-oriented-language

## Compiling and Running Are Separate Steps

In "Writing and Running a Rust Program", we showed you how to run a newly
created program. We'll break that process down and examine each step now.

Before running a Rust program, you have to compile it. You can use the Rust
compiler by entering the `rustc` command and passing it the name of your source
file, like this:

```bash
$ rustc main.rs
```

If you come from a C or C++ background, you'll notice that this is similar to
`gcc` or `clang`. After compiling successfully, Rust should output a binary
executable, which you can see on Linux or macOS by entering the `ls` command in
your shell as follows:

```bash
$ ls
main  main.rs
```

On Windows, you'd enter:

```bash
$ dir
main.exe
main.rs
```

This shows we have two files: the source code, with an `.rs` extension, and the
executable (`main.exe` on Windows, `main` everywhere else). All that's left to
do from here is run the `main` or `main.exe` file, like this:

```bash
$ ./main  # or .\main.exe on Windows
```

If *main.rs* were your "Hello, world!" program, this would print `Hello,
world!` to your terminal.

If you come from a dynamic language like Ruby, Python, or JavaScript, you may
not be used to compiling and running a program being separate steps. Rust is an
*ahead-of-time compiled* language, which means that you can compile a program,
give it to someone else, and they can run it even without Rust installed. If
you give someone a `.rb` or `.py` or `.js` file, on the other hand, they need
to have a Ruby, Python, or JavaScript implementation installed (respectively),
but you only need one command to both compile and run your program. Everything
is a tradeoff in language design.

Just compiling with `rustc` is fine for simple programs, but as your project
grows, you'll want to be able to manage all of the options your project has,
and make it easy to share your code with other people and projects. Next, I'll
introduce you to a tool called Cargo, which will help you write real-world Rust
programs.

# Hello, Cargo!

Cargo is Rust’s build system and package manager, and Rustaceans use Cargo to
manage their Rust projects. Cargo manages three things: building your code,
downloading the libraries your code depends on, and building those libraries.
We call libraries your code needs ‘dependencies’ since your code depends on
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
2. Get rid of the old executable (`main.exe` on Windows, `main` everywhere
   else).
3. Make a Cargo configuration file.

Let's get started!

### Creating a Source Directory and Removing the Old Executable

First, go back to your terminal, move to your *hello_world* directory, and
enter the following commands:

```bash
$ mkdir src
$ mv main.rs src/main.rs # or 'move main.rs src/main.rs' on Windows
$ rm main  # or 'del main.exe' on Windows
```

Cargo expects your source files to live inside a *src* directory, so do that
first. This leaves the top-level project directory (in this case,
*hello_world*) for READMEs, license information, and anything else not related
to your code. In this way, using Cargo helps you keep your projects nice and
tidy. There's a place for everything, and everything is in its place.

Now, move *main.rs* into the *src* directory, and delete the compiled file you
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
similar to INI, but has some extra goodies, and is used as Cargo’s
configuration format.

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
sections, but for now, we only have the package configuration.

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

The `run` command comes in handy when you need to rapidly iterate on a
project.

Notice that this example didn’t re-build the project. Cargo figured out that
the file hasn’t changed, and so it just ran the binary. If you'd modified your
source code, Cargo would have rebuilt the project before running it, and you
would have seen something like this:

```bash
$ cargo run
   Compiling hello_world v0.0.1 (file:///home/yourname/projects/hello_world)
     Running `target/debug/hello_world`
Hello, world!
```

Cargo checks to see if any of your project’s files have been modified, and only
rebuilds your project if they’ve changed since the last time you built it.

With simple projects, Cargo doesn't bring a whole lot over just using `rustc`,
but it will become useful in the future. This is especially true when you start
using crates; these are synonymous with a ‘library’ or ‘package’ in other
programming languages. For complex projects composed of multiple crates, it’s
much easier to let Cargo coordinate the build. Using Cargo, you can run `cargo
build`, and it should work the right way.

### Building for Release

When your project is ready for release, you can use `cargo build
--release` to compile your project with optimizations. These optimizations make
your Rust code run faster, but turning them on makes your program take longer
to compile. This is why there are two different profiles, one for development,
and one for building the final program you’ll give to a user.

### What Is That `Cargo.lock`?

Running `cargo build` also causes Cargo to create a new file called
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
*src* directory with a *main.rs* file inside. These should look familiar,
they’re exactly what we created by hand, above.

This output is all you need to get started. First, open `Cargo.toml`. It should
look something like this:

```toml
[package]

name = "hello_world"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]

[dependencies]
```

Do not worry about the `[dependencies]` line, we will come back to it later.

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

You have two options: Dive into a project with ‘[Tutorial: Guessing Game][guessinggame]’, or
start from the bottom and work your way up with ‘[Syntax and
Semantics][syntax]’. More experienced systems programmers will probably prefer
‘Tutorial: Guessing Game’, while those from dynamic backgrounds may enjoy either. Different
people learn differently! Choose whatever’s right for you.

[guessinggame]: guessing-game.html
[syntax]: syntax-and-semantics.html
