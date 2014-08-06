% The Rust Guide

<div style="border: 2px solid red; padding:5px;">
This guide is a work in progress. Until it is ready, we highly recommend that
you read the <a href="tutorial.html">Tutorial</a> instead. This work-in-progress Guide is being
displayed here in line with Rust's open development policy. Please open any
issues you find as usual.
</div>

# Welcome!

Hey there! Welcome to the Rust guide. This is the place to be if you'd like to
learn how to program in Rust. Rust is a systems programming language with a
focus on "high-level, bare-metal programming": the lowest level control a
programming language can give you, but with zero-cost, higher level
abstractions, because people aren't computers. We really think Rust is
something special, and we hope you do too.

To show you how to get going with Rust, we're going to write the traditional
"Hello, World!" program. Next, we'll introduce you to a tool that's useful for
writing real-world Rust programs and libraries: "Cargo." After that, we'll talk
about the basics of Rust, write a little program to try them out, and then learn
more advanced things.

Sound good? Let's go!

# Installing Rust

The first step to using Rust is to install it! There are a number of ways to
install Rust, but the easiest is to use the the `rustup` script. If you're on
Linux or a Mac, all you need to do is this (note that you don't need to type
in the `$`s, they just indicate the start of each command):

```{ignore}
$ curl -s http://www.rust-lang.org/rustup.sh | sudo sh
```

(If you're concerned about `curl | sudo sh`, please keep reading. Disclaimer
below.)

If you're on Windows, please [download this .exe and run
it](http://static.rust-lang.org/dist/rust-nightly-install.exe).

If you decide you don't want Rust anymore, we'll be a bit sad, but that's okay.
Not every programming language is great for everyone. Just pass an argument to
the script:

```{ignore}
$ curl -s http://www.rust-lang.org/rustup.sh | sudo sh -s -- --uninstall
```

If you used the Windows installer, just re-run the `.exe` and it will give you
an uninstall option.

You can re-run this script any time you want to update Rust. Which, at this
point, is often. Rust is still pre-1.0, and so people assume that you're using
a very recent Rust.

This brings me to one other point: some people, and somewhat rightfully so, get
very upset when we tell you to `curl | sudo sh`. And they should be! Basically,
when you do this, you are trusting that the good people who maintain Rust
aren't going to hack your computer and do bad things. That's a good instinct!
If you're one of those people, please check out the documentation on [building
Rust from Source](https://github.com/rust-lang/rust#building-from-source), or
[the official binary downloads](http://www.rust-lang.org/install.html). And we
promise that this method will not be the way to install Rust forever: it's just
the easiest way to keep people updated while Rust is in its alpha state.

Oh, we should also mention the officially supported platforms:

* Windows (7, 8, Server 2008 R2), x86 only
* Linux (2.6.18 or later, various distributions), x86 and x86-64
* OSX 10.7 (Lion) or greater, x86 and x86-64

We extensively test Rust on these platforms, and a few others, too, like
Android. But these are the ones most likely to work, as they have the most
testing.

Finally, a comment about Windows. Rust considers Windows to be a first-class
platform upon release, but if we're honest, the Windows experience isn't as
integrated as the Linux/OS X experience is. We're working on it! If anything
does not work, it is a bug. Please let us know if that happens. Each and every
commit is tested against Windows just like any other platform.

If you've got Rust installed, you can open up a shell, and type this:

```{ignore}
$ rustc --version
```

You should see some output that looks something like this:

```{ignore}
rustc 0.12.0-pre (443a1cd 2014-06-08 14:56:52 -0700)
```

If you did, Rust has been installed successfully! Congrats!

If not, there are a number of places where you can get help. The easiest is
[the #rust IRC channel on irc.mozilla.org](irc://irc.mozilla.org/#rust), which
you can access through
[Mibbit](http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust). Click
that link, and you'll be chatting with other Rustaceans (a silly nickname we
call ourselves), and we can help you out. Other great resources include [our
mailing list](https://mail.mozilla.org/listinfo/rust-dev), [the /r/rust
subreddit](http://www.reddit.com/r/rust), and [Stack
Overflow](http://stackoverflow.com/questions/tagged/rust).

# Hello, world!

Now that you have Rust installed, let's write your first Rust program. It's
traditional to make your first program in any new language one that prints the
text "Hello, world!" to the screen. The nice thing about starting with such a
simple program is that you can verify that your compiler isn't just installed,
but also working properly. And printing information to the screen is a pretty
common thing to do.

The first thing that we need to do is make a file to put our code in. I like
to make a projects directory in my home directory, and keep all my projects
there. Rust does not care where your code lives.

This actually leads to one other concern we should address: this tutorial will
assume that you have basic familiarity with the command-line. Rust does not
require that you know a whole ton about the command line, but until the
language is in a more finished state, IDE support is spotty. Rust makes no
specific demands on your editing tooling, or where your code lives.

With that said, let's make a directory in our projects directory.

```{bash}
$ mkdir ~/projects
$ cd ~/projects
$ mkdir hello_world
$ cd hello_world
```

If you're on Windows and not using PowerShell, the `~` may not work. Consult
the documentation for your shell for more details.

Let's make a new source file next. I'm going to use the syntax `editor
filename` to represent editing a file in these examples, but you should use
whatever method you want. We'll call our file `hello_world.rs`:

```{bash}
$ editor hello_world.rs
```

Rust files always end in a `.rs` extension. If you're using more than one word
in your file name, use an underscore. `hello_world.rs` versus `goodbye.rs`.

Now that you've got your file open, type this in:

```
fn main() {
    println!("Hello, world!");
}
```

Save the file, and then type this into your terminal window:

```{bash}
$ rustc hello_world.rs
$ ./hello_world # or hello_world.exe on Windows
Hello, world!
```

Success! Let's go over what just happened in detail.

```
fn main() {

}
```

These two lines define a **function** in Rust. The `main` function is special:
it's the beginning of every Rust program. The first line says "I'm declaring a
function named `main`, which takes no arguments and returns nothing." If there
were arguments, they would go inside the parentheses (`(` and `)`), and because
we aren't returning anything from this function, we've dropped that notation
entirely.  We'll get to it later.

You'll also note that the function is wrapped in curly braces (`{` and `}`).
Rust requires these around all function bodies. It is also considered good
style to put the opening curly brace on the same line as the function
declaration, with one space in between.

Next up is this line:

```
    println!("Hello, world!");
```

This line does all of the work in our little program. There are a number of
details that are important here. The first is that it's indented with four
spaces, not tabs. Please configure your editor of choice to insert four spaces
with the tab key. We provide some sample configurations for various editors
[here](https://github.com/rust-lang/rust/tree/master/src/etc).

The second point is the `println!()` part. This is calling a Rust **macro**,
which is how metaprogramming is done in Rust. If it were a function instead, it
would look like this: `println()`. For our purposes, we don't need to worry
about this difference. Just know that sometimes, you'll see a `!`, and that
means that you're calling a macro instead of a normal function. One last thing
to mention: Rust's macros are significantly different than C macros, if you've
used those. Don't be scared of using macros. We'll get to the details
eventually, you'll just have to trust us for now.

Next, `"Hello, world!"` is a **string**. Strings are a surprisingly complicated
topic in a systems programming language, and this is a **statically allocated**
string. We will talk more about different kinds of allocation later. We pass
this string as an argument to `println!`, which prints the string to the
screen. Easy enough!

Finally, the line ends with a semicolon (`;`). Rust is an **expression
oriented** language, which means that most things are expressions. The `;` is
used to indicate that this expression is over, and the next one is ready to
begin. Most lines of Rust code end with a `;`. We will cover this in-depth
later in the tutorial.

Finally, actually **compiling** and **running** our program. We can compile
with our compiler, `rustc`, by passing it the name of our source file:

```{bash}
$ rustc hello_world.rs
```

This is similar to `gcc` or `clang`, if you come from a C or C++ background. Rust
will output a binary executable. You can see it with `ls`:

```{bash}
$ ls
hello_world  hello_world.rs
```

Or on Windows:

```{bash}
$ dir
hello_world.exe  hello_world.rs
```

There are now two files: our source code, with the `.rs` extension, and the
executable (`hello_world.exe` on Windows, `hello_world` everywhere else)

```{bash}
$ ./hello_world  # or hello_world.exe on Windows
```

This prints out our `Hello, world!` text to our terminal.

If you come from a dynamically typed language like Ruby, Python, or JavaScript,
you may not be used to these two steps being separate. Rust is an
**ahead-of-time compiled language**, which means that you can compile a
program, give it to someone else, and they don't need to have Rust installed.
If you give someone a `.rb` or `.py` or `.js` file, they need to have
Ruby/Python/JavaScript installed, but you just need one command to both compile
and run your program. Everything is a tradeoff in language design, and Rust has
made its choice.

Congratulations! You have officially written a Rust program. That makes you a
Rust programmer! Welcome.

Next, I'd like to introduce you to another tool, Cargo, which is used to write
real-world Rust programs. Just using `rustc` is nice for simple things, but as
your project grows, you'll want something to help you manage all of the options
that it has, and to make it easy to share your code with other people and
projects.

# Hello, Cargo!

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

Let's convert Hello World to Cargo. The first thing we need to do to begin
using Cargo is to install Cargo. Luckily for us, the script we ran to install
Rust includes Cargo by default. If you installed Rust some other way, you may
want to [check the Cargo
README](https://github.com/rust-lang/cargo#installing-cargo-from-nightlies)
for specific instructions about installing it.

To Cargo-ify our project, we need to do two things: Make a `Cargo.toml`
configuration file, and put our source file in the right place. Let's
do that part first:

```{bash}
$ mkdir src
$ mv hello_world.rs src/hello_world.rs
```

Cargo expects your source files to live inside a `src` directory. That leaves
the top level for other things, like READMEs, licence information, and anything
not related to your code. Cargo helps us keep our projects nice and tidy. A
place for everything, and everything in its place.

Next, our configuration file:

```{bash}
$ editor Cargo.toml
```

Make sure to get this name right: you need the capital `C`!

Put this inside:

```{ignore}
[package]

name = "hello_world"
version = "0.1.0"
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
   Compiling hello_world v0.1.0 (file:/home/yourname/projects/hello_world)
$ ./target/hello_world
Hello, world!
```

Bam! We build our project with `cargo build`, and run it with
`./target/hello_world`. This hasn't bought us a whole lot over our simple use
of `rustc`, but think about the future: when our project has more than one
file, we would need to call `rustc` twice, and pass it a bunch of options to
tell it to build everything together. With Cargo, as our project grows, we can
just `cargo build` and it'll work the right way.

You'll also notice that Cargo has created a new file: `Cargo.lock`.

```{ignore,notrust}
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

# Variable bindings

The first thing we'll learn about are 'variable bindings.' They look like this:

```{rust}
let x = 5i;
```

In many languages, this is called a 'variable.' But Rust's variable bindings
have a few tricks up their sleeves. Rust has a very powerful feature called
'pattern matching' that we'll get into detail with later, but the left
hand side of a `let` expression is a full pattern, not just a variable name.
This means we can do things like:

```{rust}
let (x, y) = (1i, 2i);
```

After this expression is evaluated, `x` will be one, and `y` will be two.
Patterns are really powerful, but this is about all we can do with them so far.
So let's just keep this in the back of our minds as we go forward.

By the way, in these examples, `i` indicates that the number is an integer.

Rust is a statically typed language, which means that we specify our types up
front. So why does our first example compile? Well, Rust has this thing called
"[Hindley-Milner type
inference](http://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system)",
named after some really smart type theorists. If you clicked that link, don't
be scared: what this means for you is that Rust will attempt to infer the types
in your program, and it's pretty good at it. If it can infer the type, Rust
doesn't require you to actually type it out.

We can add the type if we want to. Types come after a colon (`:`):

```{rust}
let x: int = 5;
```

If I asked you to read this out loud to the rest of the class, you'd say "`x`
is a binding with the type `int` and the value `five`."

By default, bindings are **immutable**. This code will not compile:

```{ignore}
let x = 5i;
x = 10i;
```

It will give you this error:

```{ignore,notrust}
error: re-assignment of immutable variable `x`
     x = 10i;
     ^~~~~~~
```

If you want a binding to be mutable, you can use `mut`:

```{rust}
let mut x = 5i;
x = 10i;
```

There is no single reason that bindings are immutable by default, but we can
think about it through one of Rust's primary focuses: safety. If you forget to
say `mut`, the compiler will catch it, and let you know that you have mutated
something you may not have cared to mutate. If bindings were mutable by
default, the compiler would not be able to tell you this. If you _did_ intend
mutation, then the solution is quite easy: add `mut`.

There are other good reasons to avoid mutable state when possible, but they're
out of the scope of this guide. In general, you can often avoid explicit
mutation, and so it is preferable in Rust. That said, sometimes, mutation is
what you need, so it's not verboten.

Let's get back to bindings. Rust variable bindings have one more aspect that
differs from other languages: bindings are required to be initialized with a
value before you're allowed to use it. If we try...

```{ignore}
let x;
```

...we'll get an error:

```{ignore}
src/hello_world.rs:2:9: 2:10 error: cannot determine a type for this local variable: unconstrained type
src/hello_world.rs:2     let x;
                             ^
```

Giving it a type will compile, though:

```{ignore}
let x: int;
```

Let's try it out. Change your `src/hello_world.rs` file to look like this:

```{rust}
fn main() {
    let x: int;

    println!("Hello world!");
}
```

You can use `cargo build` on the command line to build it. You'll get a warning,
but it will still print "Hello, world!":

```{ignore,notrust}
   Compiling hello_world v0.1.0 (file:/home/you/projects/hello_world)
src/hello_world.rs:2:9: 2:10 warning: unused variable: `x`, #[warn(unused_variable)] on by default
src/hello_world.rs:2     let x: int;
                             ^
```

Rust warns us that we never use the variable binding, but since we never use it,
no harm, no foul. Things change if we try to actually use this `x`, however. Let's
do that. Change your program to look like this:

```{rust,ignore}
fn main() {
    let x: int;

    println!("The value of x is: {}", x);
}
```

And try to build it. You'll get an error:

```{bash}
$ cargo build
   Compiling hello_world v0.1.0 (file:/home/you/projects/hello_world)
src/hello_world.rs:4:39: 4:40 error: use of possibly uninitialized variable: `x`
src/hello_world.rs:4     println!("The value of x is: {}", x);
                                                           ^
note: in expansion of format_args!
<std macros>:2:23: 2:77 note: expansion site
<std macros>:1:1: 3:2 note: in expansion of println!
src/hello_world.rs:4:5: 4:42 note: expansion site
error: aborting due to previous error
Could not execute process `rustc src/hello_world.rs --crate-type bin --out-dir /home/you/projects/hello_world/target -L /home/you/projects/hello_world/target -L /home/you/projects/hello_world/target/deps` (status=101)
```

Rust will not let us use a value that has not been initialized. So why let us
declare a binding without initializing it? You'd think our first example would
have errored. Well, Rust is smarter than that. Before we get to that, let's talk
about this stuff we've added to `println!`.

If you include two curly braces (`{}`, some call them moustaches...) in your
string to print, Rust will interpret this as a request to interpolate some sort
of value. **String interpolation** is a computer science term that means "stick
in the middle of a string." We add a comma, and then `x`, to indicate that we
want `x` to be the value we're interpolating. The comma is used to separate
arguments we pass to functions and macros, if you're passing more than one.

When you just use the double curly braces, Rust will attempt to display the
value in a meaningful way by checking out its type. If you want to specify the
format in a more detailed manner, there are a [wide number of options
available](/std/fmt/index.html). For now, we'll just stick to the default:
integers aren't very complicated to print.

So, we've cleared up all of the confusion around bindings, with one exception:
why does Rust let us declare a variable binding without an initial value if we
must initialize the binding before we use it? And how does it know that we have
or have not initialized the binding? For that, we need to learn our next
concept: `if`.

# If

Rust's take on `if` is not particularly complex, but it's much more like the
`if` you'll find in a dynamically typed language than in a more traditional
systems language. So let's talk about it, to make sure you grasp the nuances.

`if` is a specific form of a more general concept, the 'branch.' The name comes
from a branch in a tree: a decision point, where depending on a choice,
multiple paths can be taken.

In the case of `if`, there is one choice that leads down two paths:

```rust
let x = 5i;

if x == 5i {
    println!("x is five!");
}
```

If we changed the value of `x` to something else, this line would not print.
More specifically, if the expression after the `if` evaluates to `true`, then
the block is executed. If it's `false`, then it is not.

If you want something to happen in the `false` case, use an `else`:

```
let x = 5i;

if x == 5i {
    println!("x is five!");
} else {
    println!("x is not five :(");
}
```

This is all pretty standard. However, you can also do this:


```
let x = 5i;

let y = if x == 5i {
    10i
} else {
    15i
};
```

Which we can (and probably should) write like this:

```
let x = 5i;

let y = if x == 5i { 10i } else { 15i };
```

This reveals two interesting things about Rust: it is an expression-based
language, and semicolons are different than in other 'curly brace and
semicolon'-based languages. These two things are related.

## Expressions vs. Statements

Rust is primarily an expression based language. There are only two kinds of
statements, and everything else is an expression.

So what's the difference? Expressions return a value, and statements do not.
In many languages, `if` is a statement, and therefore, `let x = if ...` would
make no sense. But in Rust, `if` is an expression, which means that it returns
a value. We can then use this value to initialize the binding.

Speaking of which, bindings are a kind of the first of Rust's two statements.
The proper name is a **declaration statement**. So far, `let` is the only kind
of declaration statement we've seen. Let's talk about that some more.

In some languages, variable bindings can be written as expressions, not just
statements. Like Ruby:

```{ruby}
x = y = 5
```

In Rust, however, using `let` to introduce a binding is _not_ an expression. The
following will produce a compile-time error:

```{ignore}
let x = (let y = 5i); // found `let` in ident position
```

The compiler is telling us here that it was expecting to see the beginning of
an expression, and a `let` can only begin a statement, not an expression.

Note that assigning to an already-bound variable (e.g. `y = 5i`) is still an
expression, although its value is not particularly useful. Unlike C, where an
assignment evaluates to the assigned value (e.g. `5i` in the previous example),
in Rust the value of an assignment is the unit type `()` (which we'll cover later).

The second kind of statement in Rust is the **expression statement**. Its
purpose is to turn any expression into a statement. In practical terms, Rust's
grammar expects statements to follow other statements. This means that you use
semicolons to separate expressions from each other. This means that Rust
looks a lot like most other languages that require you to use semicolons
at the end of every line, and you will see semicolons at the end of almost
every line of Rust code you see.

What is this exception that makes us say 'almost?' You saw it already, in this
code:

```
let x = 5i;

let y: int = if x == 5i { 10i } else { 15i };
```

Note that I've added the type annotation to `y`, to specify explicitly that I
want `y` to be an integer.

This is not the same as this, which won't compile:

```{ignore}
let x = 5i;

let y: int = if x == 5 { 10i; } else { 15i; };
```

Note the semicolons after the 10 and 15. Rust will give us the following error:

```{ignore,notrust}
error: mismatched types: expected `int` but found `()` (expected int but found ())
```

We expected an integer, but we got `()`. `()` is pronounced 'unit', and is a
special type in Rust's type system. `()` is different than `null` in other
languages, because `()` is distinct from other types. For example, in C, `null`
is a valid value for a variable of type `int`. In Rust, `()` is _not_ a valid
value for a variable of type `int`. It's only a valid value for variables of
the type `()`, which aren't very useful. Remember how we said statements don't
return a value? Well, that's the purpose of unit in this case. The semicolon
turns any expression into a statement by throwing away its value and returning
unit instead.

There's one more time in which you won't see a semicolon at the end of a line
of Rust code. For that, we'll need our next concept: functions.

# Functions

You've already seen one function so far, the `main` function:

```{rust}
fn main() {
}
```

This is the simplest possible function declaration. As we mentioned before,
`fn` says 'this is a function,' followed by the name, some parenthesis because
this function takes no arguments, and then some curly braces to indicate the
body. Here's a function named `foo`:

```{rust}
fn foo() {
}
```

So, what about taking arguments? Here's a function that prints a number:

```{rust}
fn print_number(x: int) {
    println!("x is: {}", x);
}
```

Here's a complete program that uses `print_number`:

```{rust}
fn main() {
    print_number(5);
}

fn print_number(x: int) {
    println!("x is: {}", x);
}
```

As you can see, function arguments work very similar to `let` declarations:
you add a type to the argument name, after a colon.

Here's a complete program that adds two numbers together and prints them:

```{rust}
fn main() {
    print_sum(5, 6);
}

fn print_sum(x: int, y: int) {
    println!("sum is: {}", x + y);
}
```

You separate arguments with a comma, both when you call the function, as well
as when you declare it.

Unlike `let`, you _must_ declare the types of function arguments. This does
not work:

```{ignore}
fn print_number(x, y) {
    println!("x is: {}", x + y);
}
```

You get this error:

```{ignore,notrust}
hello.rs:5:18: 5:19 error: expected `:` but found `,`
hello.rs:5 fn print_number(x, y) {
```

This is a deliberate design decision. While full-program inference is possible,
languages which have it, like Haskell, often suggest that documenting your
types explicitly is a best-practice. We agree that forcing functions to declare
types while allowing for inference inside of function bodies is a wonderful
compromise between full inference and no inference.

What about returning a value? Here's a function that adds one to an integer:

```{rust}
fn add_one(x: int) -> int {
    x + 1
}
```

Rust functions return exactly one value, and you declare the type after an
'arrow', which is a dash (`-`) followed by a greater-than sign (`>`).

You'll note the lack of a semicolon here. If we added it in:

```{ignore}
fn add_one(x: int) -> int {
    x + 1;
}
```

We would get an error:

```{ignore,notrust}
error: not all control paths return a value
fn add_one(x: int) -> int {
     x + 1;
}

note: consider removing this semicolon:
     x + 1;
          ^
```

Remember our earlier discussions about semicolons and `()`? Our function claims
to return an `int`, but with a semicolon, it would return `()` instead. Rust
realizes this probably isn't what we want, and suggests removing the semicolon.

This is very much like our `if` statement before: the result of the block
(`{}`) is the value of the expression. Other expression-oriented languages,
such as Ruby, work like this, but it's a bit unusual in the systems programming
world. When people first learn about this, they usually assume that it
introduces bugs. But because Rust's type system is so strong, and because unit
is its own unique type, we have never seen an issue where adding or removing a
semicolon in a return position would cause a bug.

But what about early returns? Rust does have a keyword for that, `return`:

```{rust}
fn foo(x: int) -> int {
    if x < 5 { return x; }

    x + 1
}
```

Using a `return` as the last line of a function works, but is considered poor
style:

```{rust}
fn foo(x: int) -> int {
    if x < 5 { return x; }

    return x + 1;
}
```

There are some additional ways to define functions, but they involve features
that we haven't learned about yet, so let's just leave it at that for now.


# Comments

Now that we have some functions, it's a good idea to learn about comments.
Comments are notes that you leave to other programmers to help explain things
about your code. The compiler mostly ignores them.

Rust has two kinds of comments that you should care about: **line comment**s
and **doc comment**s.

```{rust}
// Line comments are anything after '//' and extend to the end of the line.

let x = 5i; // this is also a line comment.

// If you have a long explanation for something, you can put line comments next
// to each other. Put a space between the // and your comment so that it's
// more readable.
```

The other kind of comment is a doc comment. Doc comments use `///` instead of
`//`, and support Markdown notation inside:

```{rust}
/// `hello` is a function that prints a greeting that is personalized based on
/// the name given.
///
/// # Arguments
///
/// * `name` - The name of the person you'd like to greet.
///
/// # Example
///
/// ```rust
/// let name = "Steve";
/// hello(name); // prints "Hello, Steve!"
/// ```
fn hello(name: &str) {
    println!("Hello, {}!", name);
}
```

When writing doc comments, adding sections for any arguments, return values,
and providing some examples of usage is very, very helpful.

You can use the `rustdoc` tool to generate HTML documentation from these doc
comments. We will talk more about `rustdoc` when we get to modules, as
generally, you want to export documentation for a full module.

# Compound Data Types

Rust, like many programming languages, has a number of different data types
that are built-in. You've already done some simple work with integers and
strings, but next, let's talk about some more complicated ways of storing data.

## Tuples

The first compound data type we're going to talk about are called **tuple**s.
Tuples are an ordered list of a fixed size. Like this:

```rust
let x = (1i, "hello");
```

The parenthesis and commas form this two-length tuple. Here's the same code, but
with the type annotated:

```rust
let x: (int, &str) = (1, "hello");
```

As you can see, the type of a tuple looks just like the tuple, but with each
position having a type name rather than the value. Careful readers will also
note that tuples are heterogeneous: we have an `int` and a `&str` in this tuple.
You haven't seen `&str` as a type before, and we'll discuss the details of
strings later. In systems programming languages, strings are a bit more complex
than in other languages. For now, just read `&str` as "a string slice," and
we'll learn more soon.

You can access the fields in a tuple through a **destructuring let**. Here's
an example:

```rust
let (x, y, z) = (1i, 2i, 3i);

println!("x is {}", x);
```

Remember before when I said the left hand side of a `let` statement was more
powerful than just assigning a binding? Here we are. We can put a pattern on
the left hand side of the `let`, and if it matches up to the right hand side,
we can assign multiple bindings at once. In this case, `let` 'destructures,'
or 'breaks up,' the tuple, and assigns the bits to three bindings.

This pattern is very powerful, and we'll see it repeated more later.

The last thing to say about tuples is that they are only equivalent if
the arity, types, and values are all identical.

```rust
let x = (1i, 2i, 3i);
let y = (2i, 3i, 4i);

if x == y {
    println!("yes");
} else {
    println!("no");
}
```

This will print `no`, as the values aren't equal.

One other use of tuples is to return multiple values from a function:

```rust
fn next_two(x: int) -> (int, int) { (x + 1i, x + 2i) }

fn main() {
    let (x, y) = next_two(5i);
    println!("x, y = {}, {}", x, y);
}
```

Even though Rust functions can only return one value, a tuple _is_ one value,
that happens to be made up of two. You can also see in this example how you
can destructure a pattern returned by a function, as well.

Tuples are a very simple data structure, and so are not often what you want.
Let's move on to their bigger sibling, structs.

## Structs

A struct is another form of a 'record type,' just like a tuple. There's a
difference: structs give each element that they contain a name, called a
'field' or a 'member.' Check it out:

```rust
struct Point {
    x: int,
    y: int,
}

fn main() {
    let origin = Point { x: 0i, y:  0i };

    println!("The origin is at ({}, {})", origin.x, origin.y);
}
```

There's a lot going on here, so let's break it down. We declare a struct with
the `struct` keyword, and then with a name. By convention, structs begin with a
capital letter and are also camel cased: `PointInSpace`, not `Point_In_Space`.

We can create an instance of our struct via `let`, as usual, but we use a `key:
value` style syntax to set each field. The order doesn't need to be the same as
in the original declaration.

Finally, because fields have names, we can access the field through dot
notation: `origin.x`.

The values in structs are immutable, like other bindings in Rust. However, you
can use `mut` to make them mutable:

```rust
struct Point {
    x: int,
    y: int,
}

fn main() {
    let mut point = Point { x: 0i, y:  0i };

    point.x = 5;

    println!("The point is at ({}, {})", point.x, point.y);
}
```

This will print `The point is at (5, 0)`.

## Tuple Structs and Newtypes

Rust has another data type that's like a hybrid between a tuple and a struct,
called a **tuple struct**. Tuple structs do have a name, but their fields
don't:


```
struct Color(int, int, int);
struct Point(int, int, int);
```

These two will not be equal, even if they have the same values:

```{rust,ignore}
let black  = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

It is almost always better to use a struct than a tuple struct. We would write
`Color` and `Point` like this instead:

```rust
struct Color {
    red: int,
    blue: int,
    green: int,
}

struct Point {
    x: int,
    y: int,
    z: int,
}
```

Now, we have actual names, rather than positions. Good names are important,
and with a struct, we have actual names.

There _is_ one case when a tuple struct is very useful, though, and that's a
tuple struct with only one element. We call this a 'newtype,' because it lets
you create a new type that's a synonym for another one:

```
struct Inches(int);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As you can see here, you can extract the inner integer type through a
destructuring `let`.

## Enums

Finally, Rust has a "sum type", an **enum**. Enums are an incredibly useful
feature of Rust, and are used throughout the standard library. Enums look
like this:

```
enum Ordering {
    Less,
    Equal,
    Greater,
}
```

This is an enum that is provided by the Rust standard library. An `Ordering`
can only be _one_ of `Less`, `Equal`, or `Greater` at any given time. Here's
an example:

```rust
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    let ordering = cmp(x, y);

    if ordering == Less {
        println!("less");
    } else if ordering == Greater {
        println!("greater");
    } else if ordering == Equal {
        println!("equal");
    }
}
```

`cmp` is a function that compares two things, and returns an `Ordering`. We
return either `Less`, `Greater`, or `Equal`, depending on if the two values
are greater, less, or equal.

The `ordering` variable has the type `Ordering`, and so contains one of the
three values. We can then do a bunch of `if`/`else` comparisons to check
which one it is.

However, repeated `if`/`else` comparisons get quite tedious. Rust has a feature
that not only makes them nicer to read, but also makes sure that you never
miss a case. Before we get to that, though, let's talk about another kind of
enum: one with values.

This enum has two variants, one of which has a value:

```{rust}
enum OptionalInt {
    Value(int),
    Missing,
}

fn main() {
    let x = Value(5);
    let y = Missing;

    match x {
        Value(n) => println!("x is {:d}", n),
        Missing  => println!("x is missing!"),
    }

    match y {
        Value(n) => println!("y is {:d}", n),
        Missing  => println!("y is missing!"),
    }
}
```

This enum represents an `int` that we may or may not have. In the `Missing`
case, we have no value, but in the `Value` case, we do. This enum is specific
to `int`s, though. We can make it usable by any type, but we haven't quite
gotten there yet!

You can have any number of values in an enum:

```
enum OptionalColor {
    Color(int, int, int),
    Missing
}
```

Enums with values are quite useful, but as I mentioned, they're even more
useful when they're generic across types. But before we get to generics, let's
talk about how to fix this big `if`/`else` statements we've been writing. We'll
do that with `match`.

# Match

Often, a simple `if`/`else` isn't enough, because you have more than two
possible options. And `else` conditions can get incredibly complicated. So
what's the solution?

Rust has a keyword, `match`, that allows you to replace complicated `if`/`else`
groupings with something more powerful. Check it out:

```rust
let x = 5i;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```

`match` takes an expression, and then branches based on its value. Each 'arm' of
the branch is of the form `val => expression`. When the value matches, that arm's
expression will be evaluated. It's called `match` because of the term 'pattern
matching,' which `match` is an implementation of.

So what's the big advantage here? Well, there are a few. First of all, `match`
does 'exhaustiveness checking.' Do you see that last arm, the one with the
underscore (`_`)? If we remove that arm, Rust will give us an error:

```{ignore,notrust}
error: non-exhaustive patterns: `_` not covered
```

In other words, Rust is trying to tell us we forgot a value. Because `x` is an
integer, Rust knows that it can have a number of different values. For example,
`6i`. But without the `_`, there is no arm that could match, and so Rust refuses
to compile. `_` is sort of like a catch-all arm. If none of the other arms match,
the arm with `_` will. And since we have this catch-all arm, we now have an arm
for every possible value of `x`, and so our program will now compile.

`match` statements also destructure enums, as well. Remember this code from the
section on enums?

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    let ordering = cmp(x, y);

    if ordering == Less {
        println!("less");
    } else if ordering == Greater {
        println!("greater");
    } else if ordering == Equal {
        println!("equal");
    }
}
```

We can re-write this as a `match`:

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    match cmp(x, y) {
        Less    => println!("less"),
        Greater => println!("greater"),
        Equal   => println!("equal"),
    }
}
```

This version has way less noise, and it also checks exhaustively to make sure
that we have covered all possible variants of `Ordering`. With our `if`/`else`
version, if we had forgotten the `Greater` case, for example, our program would
have happily compiled. If we forget in the `match`, it will not. Rust helps us
make sure to cover all of our bases.

`match` is also an expression, which means we can use it on the right hand side
of a `let` binding. We could also implement the previous line like this:

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    let result = match cmp(x, y) {
        Less    => "less",
        Greater => "greater",
        Equal   => "equal",
    };

    println!("{}", result);
}
```

In this case, it doesn't make a lot of sense, as we are just making a temporary
string where we don't need to, but sometimes, it's a nice pattern.

# Looping

Looping is the last basic construct that we haven't learned yet in Rust. Rust has
two main looping constructs: `for` and `while`.

## `for`

The `for` loop is used to loop a particular number of times. Rust's `for` loops
work a bit differently than in other systems languages, however. Rust's `for`
loop doesn't look like this C `for` loop:

```{ignore,c}
for (x = 0; x < 10; x++) {
    printf( "%d\n", x );
}
```

It looks like this:

```{rust}
for x in range(0i, 10i) {
    println!("{:d}", x);
}
```

In slightly more abstract terms,

```{ignore,notrust}
for var in expression {
    code
}
```

The expression is an iterator, which we will discuss in more depth later in the
guide. The iterator gives back a series of elements. Each element is one
iteration of the loop. That value is then bound to the name `var`, which is
valid for the loop body. Once the body is over, the next value is fetched from
the iterator, and we loop another time. When there are no more values, the
`for` loop is over.

In our example, the `range` function is a function, provided by Rust, that
takes a start and an end position, and gives an iterator over those values. The
upper bound is exclusive, though, so our loop will print `0` through `9`, not
`10`.

Rust does not have the "C style" `for` loop on purpose. Manually controlling
each element of the loop is complicated and error prone, even for experienced C
developers. There's an old joke that goes, "There are two hard problems in
computer science: naming things, cache invalidation, and off-by-one errors."
The joke, of course, being that the setup says "two hard problems" but then
lists three things. This happens quite a bit with "C style" `for` loops.

We'll talk more about `for` when we cover **vector**s, later in the Guide.

## `while`

The other kind of looping construct in Rust is the `while` loop. It looks like
this:

```{rust}
let mut x = 5u;
let mut done = false;

while !done {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { done = true; }
}
```

`while` loops are the correct choice when you're not sure how many times
you need to loop. 

If you need an infinite loop, you may be tempted to write this:

```{rust,ignore}
while true {
```

Rust has a dedicated keyword, `loop`, to handle this case:

```{rust,ignore}
loop {
```

Rust's control-flow analysis treats this construct differently than a
`while true`, since we know that it will always loop. The details of what
that _means_ aren't super important to understand at this stage, but in
general, the more information we can give to the compiler, the better it
can do with safety and code generation. So you should always prefer
`loop` when you plan to loop infinitely.

## Ending iteration early

Let's take a look at that `while` loop we had earlier:

```{rust}
let mut x = 5u;
let mut done = false;

while !done {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { done = true; }
}
```

We had to keep a dedicated `mut` boolean variable binding, `done`, to know
when we should skip out of the loop. Rust has two keywords to help us with
modifying iteration: `break` and `continue`.

In this case, we can write the loop in a better way with `break`:

```{rust}
let mut x = 5u;

loop {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { break; }
}
```

We now loop forever with `loop`, and use `break` to break out early.

`continue` is similar, but instead of ending the loop, goes to the next
iteration: This will only print the odd numbers:

```
for x in range(0i, 10i) {
    if x % 2 == 0 { continue; }

    println!("{:d}", x);
}
```

Both `continue` and `break` are valid in both kinds of loops.

We have now learned all of the most basic Rust concepts. We're ready to start
building our guessing game, but we need to know how to do one last thing first:
get input from the keyboard. You can't have a guessing game without the ability
to guess!

# Standard Input

Getting input from the keyboard is pretty easy, but uses some things
we haven't seen before. Here's a simple program that reads some input,
and then prints it back out:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = std::io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

Let's go over these chunks, one by one:

```{rust,ignore}
std::io::stdin();
```

This calls a function, `stdin()`, that lives inside the `std::io` module. As
you can imagine, everything in `std` is provided by Rust, the 'standard
library.' We'll talk more about the module system later.

Since writing the fully qualified name all the time is annoying, we can use
the `use` statement to import it in:

```{rust}
use std::io::stdin;

stdin();
```

However, it's considered better practice to not import individual functions, but
to import the module, and only use one level of qualification:

```{rust}
use std::io;

io::stdin();
```

Let's update our example to use this style:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

Next up:

```{rust,ignore}
.read_line()
```

The `read_line()` method can be called on the result of `stdin()` to return
a full line of input. Nice and easy.

```{rust,ignore}
.ok().expect("Failed to read line");
```

Do you remember this code? 

```
enum OptionalInt {
    Value(int),
    Missing,
}

fn main() {
    let x = Value(5);
    let y = Missing;

    match x {
        Value(n) => println!("x is {:d}", n),
        Missing  => println!("x is missing!"),
    }

    match y {
        Value(n) => println!("y is {:d}", n),
        Missing  => println!("y is missing!"),
    }
}
```

We had to match each time, to see if we had a value or not. In this case,
though, we _know_ that `x` has a `Value`. But `match` forces us to handle
the `missing` case. This is what we want 99% of the time, but sometimes, we
know better than the compiler.

Likewise, `read_line()` does not return a line of input. It _might_ return a
line of input. It might also fail to do so. This could happen if our program
isn't running in a terminal, but as part of a cron job, or some other context
where there's no standard input. Because of this, `read_line` returns a type
very similar to our `OptionalInt`: an `IoResult<T>`. We haven't talked about
`IoResult<T>` yet because it is the **generic** form of our `OptionalInt`.
Until then, you can think of it as being the same thing, just for any type, not
just `int`s.

Rust provides a method on these `IoResult<T>`s called `ok()`, which does the
same thing as our `match` statement, but assuming that we have a valid value.
If we don't, it will terminate our program. In this case, if we can't get
input, our program doesn't work, so we're okay with that. In most cases, we
would want to handle the error case explicitly. The result of `ok()` has a
method, `expect()`, which allows us to give an error message if this crash
happens.

We will cover the exact details of how all of this works later in the Guide.
For now, this gives you enough of a basic understanding to work with.

Back to the code we were working on! Here's a refresher:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = io::stdin().read_line().ok().expect("Failed to read line");

    println!("{}", input);
}
```

With long lines like this, Rust gives you some flexibility with the whitespace.
We _could_ write the example like this:

```{rust,ignore}
use std::io;

fn main() {
    println!("Type something!");

    let input = io::stdin()
                  .read_line()
                  .ok()
                  .expect("Failed to read line");

    println!("{}", input);
}
```

Sometimes, this makes things more readable. Sometimes, less. Use your judgement
here.

That's all you need to get basic input from the standard input! It's not too
complicated, but there are a number of small parts.

# Guessing Game

Okay! We've got the basics of Rust down. Let's write a bigger program.

For our first project, we'll implement a classic beginner programming problem:
the guessing game. Here's how it works: Our program will generate a random
integer between one and a hundred. It will then prompt us to enter a guess.
Upon entering our guess, it will tell us if we're too low or too high. Once we
guess correctly, it will congratulate us, and print the number of guesses we've
taken to the screen. Sound good?

## Set up

Let's set up a new project. Go to your projects directory. Remember how we
had to create our directory structure and a `Cargo.toml` for `hello_world`? Cargo
has a command that does that for us. Let's give it a shot:

```{bash}
$ cd ~/projects
$ cargo new guessing_game --bin
$ cd guessing_game
```

We pass the name of our project to `cargo new`, and then the `--bin` flag,
since we're making a binary, rather than a library.

Check out the generated `Cargo.toml`:

```{ignore}
[package]

name = "guessing_game"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
```

Cargo gets this information from your environment. If it's not correct, go ahead
and fix that.

Finally, Cargo generated a hello, world for us. Check out `src/main.rs`:

```{rust}
fn main() {
    println!("Hello world!");
}
```

Let's try compiling what Cargo gave us:

```{bash}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$
```

Excellent! Open up your `src/main.rs` again. We'll be writing all of
our code in this file. We'll talk about multiple-file projects later on in the
guide.

## Processing a Guess

Let's get to it! The first thing we need to do for our guessing game is
allow our player to input a guess. Put this in your `src/main.rs`:

```{rust,no_run}
use std::io;

fn main() {
    println!("Guess the number!");

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");

    println!("You guessed: {}", input);
}
```

You've seen this code before, when we talked about standard input. We
import the `std::io` module with `use`, and then our `main` function contains
our program's logic. We print a little message announcing the game, ask the
user to input a guess, get their input, and then print it out.

Because we talked about this in the section on standard I/O, I won't go into
more details here. If you need a refresher, go re-read that section.

## Generating a secret number

Next, we need to generate a secret number. To do that, we need to use Rust's
random number generation, which we haven't talked about yet. Rust includes a
bunch of interesting functions in its standard library. If you need a bit of
code, it's possible that it's already been written for you! In this case,
we do know that Rust has random number generation, but we don't know how to
use it.

Enter the docs. Rust has a page specifically to document the standard library.
You can find that page [here](std/index.html). There's a lot of information on
that page, but the best part is the search bar. Right up at the top, there's
a box that you can enter in a search term. The search is pretty primitive
right now, but is getting better all the time. If you type 'random' in that
box, the page will update to [this
one](http://doc.rust-lang.org/std/index.html?search=random). The very first
result is a link to
[std::rand::random](http://doc.rust-lang.org/std/rand/fn.random.html). If we
click on that result, we'll be taken to its documentation page.

This page shows us a few things: the type signature of the function, some
explanatory text, and then an example. Let's modify our code to add in the
`random` function:

```{rust,ignore}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random() % 100i) + 1i;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

The first thing we changed was to `use std::rand`, as the docs
explained.  We then added in a `let` expression to create a variable binding
named `secret_number`, and we printed out its result. Let's try to compile
this using `cargo build`:

```{notrust,no_run}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
src/main.rs:7:26: 7:34 error: the type of this value must be known in this context
src/main.rs:7     let secret_number = (rand::random() % 100i) + 1i;
                                       ^~~~~~~~
error: aborting due to previous error
```

It didn't work! Rust says "the type of this value must be known in this
context." What's up with that? Well, as it turns out, `rand::random()` can
generate many kinds of random values, not just integers. And in this case, Rust
isn't sure what kind of value `random()` should generate. So we have to help
it. With number literals, we just add an `i` onto the end to tell Rust they're
integers, but that does not work with functions. There's a different syntax,
and it looks like this:

```{rust,ignore}
rand::random::<int>();
```

This says "please give me a random `int` value." We can change our code to use
this hint...

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<int>() % 100i) + 1i;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

... and then recompile:

```{notrust,ignore}
$ cargo build
  Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$
```

Excellent! Try running our new program a few times:

```{notrust,ignore}
$ ./target/guessing_game 
Guess the number!
The secret number is: 7
Please input your guess.
4
You guessed: 4
$ ./target/guessing_game 
Guess the number!
The secret number is: 83
Please input your guess.
5
You guessed: 5
$ ./target/guessing_game 
Guess the number!
The secret number is: -29
Please input your guess.
42
You guessed: 42
```

Wait. Negative 29? We wanted a number between one and a hundred! We have two
options here: we can either ask `random()` to generate an unsigned integer, which
can only be positive, or we can use the `abs()` function. Let's go with the
unsigned integer approach. If we want a random positive number, we should ask for
a random positive number. Our code looks like this now:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);
}
```

And trying it out:

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$ ./target/guessing_game 
Guess the number!
The secret number is: 57
Please input your guess.
3
You guessed: 3
```

Great! Next up: let's compare our guess to the secret guess.

## Comparing guesses

If you remember, earlier in the tutorial, we made a `cmp` function that compared
two numbers. Let's add that in, along with a `match` statement to compare the
guess to the secret guess:

```{rust,ignore}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);

    match cmp(input, secret_number) { 
        Less    => println!("Too small!"),
        Greater => println!("Too big!"),
        Equal   => { println!("You win!"); },
    }
}

fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

If we try to compile, we'll get some errors:

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
src/main.rs:20:15: 20:20 error: mismatched types: expected `int` but found `collections::string::String` (expected int but found struct collections::string::String)
src/main.rs:20     match cmp(input, secret_number) {
                             ^~~~~
src/main.rs:20:22: 20:35 error: mismatched types: expected `int` but found `uint` (expected int but found uint)
src/main.rs:20     match cmp(input, secret_number) {
                                    ^~~~~~~~~~~~~
error: aborting due to 2 previous errors
```

This often happens when writing Rust programs, and is one of Rust's greatest
strengths. You try out some code, see if it compiles, and Rust tells you that
you've done something wrong. In this case, our `cmp` function works on integers,
but we've given it unsigned integers. In this case, the fix is easy, because
we wrote the `cmp` function! Let's change it to take `uint`s:

```{rust,ignore}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");


    println!("You guessed: {}", input);

    match cmp(input, secret_number) {
        Less    => println!("Too small!"),
        Greater => println!("Too big!"),
        Equal   => { println!("You win!"); },
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

And try compiling again:

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
src/main.rs:20:15: 20:20 error: mismatched types: expected `uint` but found `collections::string::String` (expected uint but found struct collections::string::String)
src/main.rs:20     match cmp(input, secret_number) {
                             ^~~~~
error: aborting due to previous error
```

This error is similar to the last one: we expected to get a `uint`, but we got
a `String` instead! That's because our `input` variable is coming from the
standard input, and you can guess anything. Try it:

```{notrust,ignore}
$ ./target/guessing_game 
Guess the number!
The secret number is: 73
Please input your guess.
hello
You guessed: hello
```

Oops! Also, you'll note that we just ran our program even though it didn't compile.
This works because the older version we did successfully compile was still lying
around. Gotta be careful!

Anyway, we have a `String`, but we need a `uint`. What to do? Well, there's
a function for that:

```{rust,ignore}
let input = io::stdin().read_line()
                       .ok()
                       .expect("Failed to read line");
let guess: Option<uint> = from_str(input.as_slice());
```

The `from_str` function takes in a `&str` value and converts it into something.
We tell it what kind of something with a type hint. Remember our type hint with
`random()`? It looked like this:

```{rust,ignore}
rand::random::<uint>();
```

There's an alternate way of providing a hint too, and that's declaring the type
in a `let`:

```{rust,ignore}
let x: uint = rand::random();
```

In this case, we say `x` is a `uint` explicitly, so Rust is able to properly
tell `random()` what to generate. In a similar fashion, both of these work:

```{rust,ignore}
let guess = from_str::<Option<uint>>("5");
let guess: Option<uint> = from_str("5");
```

In this case, I happen to prefer the latter, and in the `random()` case, I prefer
the former. I think the nested `<>`s make the first option especially ugly and
a bit harder to read.

Anyway, with us now convering our input to a number, our code looks like this:

```{rust,ignore}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = from_str(input.as_slice());



    println!("You guessed: {}", input_num);

    match cmp(input_num, secret_number) {
        Less    => println!("Too small!"),
        Greater => println!("Too big!"),
        Equal   => { println!("You win!"); },
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

Let's try it out!

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
src/main.rs:22:15: 22:24 error: mismatched types: expected `uint` but found `core::option::Option<uint>` (expected uint but found enum core::option::Option)
src/main.rs:22     match cmp(input_num, secret_number) {
                             ^~~~~~~~~
error: aborting due to previous error
```

Oh yeah! Our `input_num` has the type `Option<uint>`, rather than `uint`. We
need to unwrap the Option. If you remember from before, `match` is a great way
to do that. Try this code:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = from_str(input.as_slice());

    let num = match input_num {
        Some(num) => num,
        None      => {
            println!("Please input a number!");
            return;
        }
    };


    println!("You guessed: {}", num);

    match cmp(num, secret_number) {
        Less    => println!("Too small!"),
        Greater => println!("Too big!"),
        Equal   => { println!("You win!"); },
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

We use a `match` to either give us the `uint` inside of the `Option`, or we
print an error message and return. Let's give this a shot:

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$ ./target/guessing_game 
Guess the number!
The secret number is: 17
Please input your guess.
5
Please input a number!
$
```

Uh, what? But we did!

... actually, we didn't. See, when you get a line of input from `stdin()`,
you get all the input. Including the `\n` character from you pressing Enter.
So, `from_str()` sees the string `"5\n"` and says "nope, that's not a number,
there's non-number stuff in there!" Luckily for us, `&str`s have an easy
method we can use defined on them: `trim()`. One small modification, and our
code looks like this:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let input = io::stdin().read_line()
                           .ok()
                           .expect("Failed to read line");
    let input_num: Option<uint> = from_str(input.as_slice().trim());

    let num = match input_num {
        Some(num) => num,
        None      => {
            println!("Please input a number!");
            return;
        }
    };


    println!("You guessed: {}", num);

    match cmp(num, secret_number) {
        Less    => println!("Too small!"),
        Greater => println!("Too big!"),
        Equal   => { println!("You win!"); },
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

Let's try it!

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$ ./target/guessing_game 
Guess the number!
The secret number is: 58
Please input your guess.
  76  
You guessed: 76
Too big!
$
```

Nice! You can see I even added spaces before my guess, and it still figured
out that I guessed 76. Run the program a few times, and verify that guessing
the number works, as well as guessing a number too small.

The Rust compiler helped us out quite a bit there! This technique is called
"lean on the compiler," and it's often useful when working on some code. Let
the error messages help guide you towards the correct types.

Now we've got most of the game working, but we can only make one guess. Let's
change that by adding loops!

## Looping

As we already discussed, the `loop` key word gives us an infinite loop. So
let's add that in:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = from_str(input.as_slice().trim());

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                return;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Less    => println!("Too small!"),
            Greater => println!("Too big!"),
            Equal   => { println!("You win!"); },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

And try it out. But wait, didn't we just add an infinite loop? Yup. Remember
that `return`? If we give a non-number answer, we'll `return` and quit. Observe:

```{notrust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$ ./target/guessing_game 
Guess the number!
The secret number is: 59
Please input your guess.
45
You guessed: 45
Too small!
Please input your guess.
60
You guessed: 60
Too big!
Please input your guess.
59
You guessed: 59
You win!
Please input your guess.
quit
Please input a number!
$
```

Ha! `quit` actually quits. As does any other non-number input. Well, this is
suboptimal to say the least. First, let's actually quit when you win the game:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = from_str(input.as_slice().trim());

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                return;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Less    => println!("Too small!"),
            Greater => println!("Too big!"),
            Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

By adding the `return` line after the `You win!`, we'll exit the program when
we win. We have just one more tweak to make: when someone inputs a non-number,
we don't want to quit, we just want to ignore it. Change that `return` to
`continue`:


```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    println!("The secret number is: {}", secret_number);

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = from_str(input.as_slice().trim());

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                continue;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Less    => println!("Too small!"),
            Greater => println!("Too big!"),
            Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

Now we should be good! Let's try:

```{rust,ignore}
$ cargo build
   Compiling guessing_game v0.1.0 (file:/home/you/projects/guessing_game)
$ ./target/guessing_game 
Guess the number!
The secret number is: 61
Please input your guess.
10
You guessed: 10
Too small!
Please input your guess.
99
You guessed: 99
Too big!
Please input your guess.
foo
Please input a number!
Please input your guess.
61
You guessed: 61
You win!
```

Awesome! With one tiny last tweak, we have finished the guessing game. Can you
think of what it is? That's right, we don't want to print out the secret number.
It was good for testing, but it kind of ruins the game. Here's our final source:

```{rust,no_run}
use std::io;
use std::rand;

fn main() {
    println!("Guess the number!");

    let secret_number = (rand::random::<uint>() % 100u) + 1u;

    loop {

        println!("Please input your guess.");

        let input = io::stdin().read_line()
                               .ok()
                               .expect("Failed to read line");
        let input_num: Option<uint> = from_str(input.as_slice().trim());

        let num = match input_num {
            Some(num) => num,
            None      => {
                println!("Please input a number!");
                continue;
            }
        };


        println!("You guessed: {}", num);

        match cmp(num, secret_number) {
            Less    => println!("Too small!"),
            Greater => println!("Too big!"),
            Equal   => {
                println!("You win!");
                return;
            },
        }
    }
}

fn cmp(a: uint, b: uint) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}
```

## Complete!

At this point, you have successfully built the Guessing Game! Congratulations!

You've now learned the basic syntax of Rust. All of this is relatively close to
various other programming languages you have used in the past. These
fundamental syntactical and semantic elements will form the foundation for the
rest of your Rust education.

Now that you're an expert at the basics, it's time to learn about some of
Rust's more unique features.

# Crates and Modules

Rust features a strong module system, but it works a bit differently than in
other programming languages. Rust's module system has two main components:
**crate**s, and **module**s.

A crate is Rust's unit of independent compilation. Rust always compiles one
crate at a time, producing either a library or an executable. However, executables
usually depend on libraries, and many libraries depend on other libraries as well.
To support this, crates can depend on other crates.

Each crate contains a hierarchy of modules. This tree starts off with a single
module, called the **crate root**. Within the crate root, we can declare other
modules, which can contain other modules, as deeply as you'd like.

Note that we haven't mentioned anything about files yet. Rust does not impose a
particular relationship between your filesystem structure and your module
structure. That said, there is a conventional approach to how Rust looks for
modules on the file system, but it's also overrideable.

Enough talk, let's build something! Let's make a new project called `modules`.

```{bash,ignore}
$ cd ~/projects
$ cargo new modules --bin
```

Let's double check our work by compiling:

```{bash,ignore}
$ cargo build
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
$ ./target/modules
Hello, world!
```

Excellent! So, we already have a single crate here: our `src/main.rs` is a crate.
Everything in that file is in the crate root. A crate that generates an executable
defines a `main` function inside its root, as we've done here.

Let's define a new module inside our crate. Edit `src/main.rs` to look
like this:

```
fn main() {
    println!("Hello, world!");
}

mod hello {
    fn print_hello() {
        println!("Hello, world!");
    }
}
```

We now have a module named `hello` inside of our crate root. Modules use
`snake_case` naming, like functions and variable bindings.

Inside the `hello` module, we've defined a `print_hello` function. This will
also print out our hello world message. Modules allow you to split up your
program into nice neat boxes of functionality, grouping common things together,
and keeping different things apart. It's kinda like having a set of shelves:
a place for everything and everything in its place.

To call our `print_hello` function, we use the double colon (`::`):

```{rust,ignore}
hello::print_hello();
```

You've seen this before, with `io::stdin()` and `rand::random()`. Now you know
how to make your own. However, crates and modules have rules about
**visibility**, which controls who exactly may use the functions defined in a
given module. By default, everything in a module is private, which means that
it can only be used by other functions in the same module. This will not
compile:

```{rust,ignore}
fn main() {
    hello::print_hello();
}

mod hello {
    fn print_hello() {
        println!("Hello, world!");
    }
}
```

It gives an error:

```{notrust,ignore}
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
src/main.rs:2:5: 2:23 error: function `print_hello` is private
src/main.rs:2     hello::print_hello();
                  ^~~~~~~~~~~~~~~~~~
```

To make it public, we use the `pub` keyword:

```{rust}
fn main() {
    hello::print_hello();
}

mod hello {
    pub fn print_hello() {
        println!("Hello, world!");
    }
}
```

This will work:

```{notrust,ignore}
$ cargo build
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
$
```

Before we move on, let me show you one more Cargo command: `run`. `cargo run`
is kind of like `cargo build`, but it also then runs the produced exectuable.
Try it out:

```{notrust,ignore}
$ cargo run
   Compiling modules v0.1.0 (file:/home/steve/tmp/modules)
     Running `target/modules`
Hello, world!
$
```

Nice!

There's a common pattern when you're building an executable: you build both an
executable and a library, and put most of your logic in the library. That way,
other programs can use that library to build their own functionality.

Let's do that with our project. If you remember, libraries and executables
are both crates, so while our project has one crate now, let's make a second:
one for the library, and one for the executable.

To make the second crate, open up `src/lib.rs` and put this code in it:

```{rust}
mod hello {
    pub fn print_hello() {
        println!("Hello, world!");
    }
}
```

And change your `src/main.rs` to look like this:

```{rust,ignore}
extern crate modules;

fn main() {
    modules::hello::print_hello();
}
```

There's been a few changes. First, we moved our `hello` module into its own
file, `src/lib.rs`. This is the file that Cargo expects a library crate to
be named, by convention.

Next, we added an `extern crate modules` to the top of our `src/main.rs`. This,
as you can guess, lets Rust know that our crate relies on another, external
crate. We also had to modify our call to `print_hello`: now that it's in
another crate, we need to first specify the crate, then the module inside of it,
then the function name.

This doesn't _quite_ work yet. Try it:

```{notrust,ignore}
$ cargo build
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
/home/you/projects/modules/src/lib.rs:2:5: 4:6 warning: code is never used: `print_hello`, #[warn(dead_code)] on by default
/home/you/projects/modules/src/lib.rs:2     pub fn print_hello() {
/home/you/projects/modules/src/lib.rs:3         println!("Hello, world!");
/home/you/projects/modules/src/lib.rs:4     }
/home/you/projects/modules/src/main.rs:4:5: 4:32 error: function `print_hello` is private
/home/you/projects/modules/src/main.rs:4     modules::hello::print_hello();
                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
error: aborting due to previous error
Could not compile `modules`.
```

First, we get a warning that some code is never used. Odd. Next, we get an error:
`print_hello` is private, so we can't call it. Notice that the first error came
from `src/lib.rs`, and the second came from `src/main.rs`: cargo is smart enough
to build it all with one command. Also, after seeing the second error, the warning
makes sense: we never actually call `hello_world`, because we're not allowed to!

Just like modules, crates also have private visibility by default. Any modules
inside of a crate can only be used by other modules in the crate, unless they
use `pub`. In `src/lib.rs`, change this line:

```{rust,ignore}
mod hello {
```

To this:

```{rust,ignore}
pub mod hello {
```

And everything should work:

```{notrust,ignore}
$ cargo run
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
     Running `target/modules`
Hello, world!
```

Let's do one more thing: add a `goodbye` module as well. Imagine a `src/lib.rs`
that looks like this:

```{rust,ignore}
pub mod hello {
    pub fn print_hello() {
        println!("Hello, world!");
    }
}

pub mod goodbye {
    pub fn print_goodbye() {
        println!("Goodbye for now!");
    }
}
```

Now, these two modules are pretty small, but imagine we've written a real, large
program: they could both be huge. So maybe we want to move them into their own
files. We can do that pretty easily, and there are two different conventions
for doing it. Let's give each a try. First, make `src/lib.rs` look like this:

```{rust,ignore}
pub mod hello;
pub mod goodbye;
```

This tells Rust that this crate has two public modules: `hello` and `goodbye`.

Next, make a `src/hello.rs` that contains this:

```{rust,ignore}
pub fn print_hello() {
    println!("Hello, world!");
}
```

When we include a module like this, we don't need to make the `mod` declaration,
it's just understood. This helps prevent 'rightward drift': when you end up
indenting so many times that your code is hard to read.

Finally, make a new directory, `src/goodbye`, and make a new file in it,
`src/goodbye/mod.rs`:

```{rust,ignore}
pub fn print_goodbye() {
    println!("Bye for now!");
}
```

Same deal, but we can make a folder with a `mod.rs` instead of `mod_name.rs` in
the same directory. If you have a lot of modules, nested folders can make
sense.  For example, if the `goodbye` module had its _own_ modules inside of
it, putting all of that in a folder helps keep our directory structure tidy.
And in fact, if you place the modules in separate files, they're required to be
in separate folders.

This should all compile as usual:

```{notrust,ignore}
$ cargo build
   Compiling modules v0.1.0 (file:/home/you/projects/modules)
$
```

We've seen how the `::` operator can be used to call into modules, but when
we have deep nesting like `modules::hello::say_hello`, it can get tedious.
That's why we have the `use` keyword.

`use` allows us to bring certain names into another scope. For example, here's
our main program:

```{rust,ignore}
extern crate modules;

fn main() {
    modules::hello::print_hello();
}
```

We could instead write this:

```{rust,ignore}
extern crate modules;

use modules::hello::print_hello;

fn main() {
    print_hello();
}
```

By bringing `print_hello` into scope, we don't need to qualify it anymore. However,
it's considered proper style to do write this code like like this:

```{rust,ignore}
extern crate modules;

use modules::hello;

fn main() {
    hello::print_hello();
}
```

By just bringing the module into scope, we can keep one level of namespacing.

# Testing

Traditionally, testing has not been a strong suit of most systems programming
languages. Rust, however, has very basic testing built into the language
itself.  While automated testing cannot prove that your code is bug-free, it is
useful for verifying that certain behaviors work as intended.

Here's a very basic test:

```{rust}
#[test]
fn is_one_equal_to_one() {
    assert_eq!(1i, 1i);
}
```

You may notice something new: that `#[test]`. Before we get into the mechanics
of testing, let's talk about attributes.

## Attributes

Rust's testing system uses **attribute**s to mark which functions are tests.
Attributes can be placed on any Rust **item**. Remember how most things in
Rust are an expression, but `let` is not? Item declarations are also not
expressions. Here's a list of things that qualify as an item:

* functions
* modules
* type definitions
* structures
* enumerations
* static items
* traits
* implementations

You haven't learned about all of these things yet, but that's the list. As
you can see, functions are at the top of it.

Attributes can appear in three ways:

1. A single identifier, the attribute name. `#[test]` is an example of this.
2. An identifier followed by an equals sign (`=`) and a literal. `#[cfg=test]`
   is an example of this.
3. An identifier followed by a parenthesized list of sub-attribute arguments.
   `#[cfg(unix, target_word_size = "32")]` is an example of this, where one of
    the sub-arguments is of the second kind.

There are a number of different kinds of attributes, enough that we won't go
over them all here. Before we talk about the testing-specific attributes, I
want to call out one of the most important kinds of attributes: stability
markers.

## Stability attributes

Rust provides six attributes to indicate the stability level of various
parts of your library. The six levels are:

* deprecated: this item should no longer be used. No guarantee of backwards
  compatibility.
* experimental: This item was only recently introduced or is otherwise in a
  state of flux. It may change significantly, or even be removed. No guarantee
  of backwards-compatibility.
* unstable: This item is still under development, but requires more testing to
  be considered stable. No guarantee of backwards-compatibility.
* stable: This item is considered stable, and will not change significantly.
  Guarantee of backwards-compatibility.
* frozen: This item is very stable, and is unlikely to change. Guarantee of
  backwards-compatibility.
* locked: This item will never change unless a serious bug is found. Guarantee
  of backwards-compatibility.

All of Rust's standard library uses these attribute markers to communicate
their relative stability, and you should use them in your code, as well.
There's an associated attribute, `warn`, that allows you to warn when you
import an item marked with certain levels: deprecated, experimental and
unstable. For now, only deprecated warns by default, but this will change once
the standard library has been stabilized.

You can use the `warn` attribute like this:

```{rust,ignore}
#![warn(unstable)]
```

And later, when you import a crate:

```{rust,ignore}
extern crate some_crate;
```

You'll get a warning if you use something marked unstable.

You may have noticed an exclamation point in the `warn` attribute declaration.
The `!` in this attribute means that this attribute applies to the enclosing
item, rather than to the item that follows the attribute. So this `warn`
attribute declaration applies to the enclosing crate itself, rather than
to whatever item statement follows it:

```{rust,ignore}
// applies to the crate we're in
#![warn(unstable)]

extern crate some_crate;

// applies to the following `fn`.
#[test]
fn a_test() {
  // ...
}
```

## Writing tests

Let's write a very simple crate in a test-driven manner. You know the drill by
now: make a new project:

```{bash,ignore}
$ cd ~/projects
$ cargo new testing --bin
$ cd testing
```

And try it out:

```{notrust,ignore}
$ cargo run
   Compiling testing v0.1.0 (file:/home/you/projects/testing)
     Running `target/testing`
Hello, world!
$
```

Great. Rust's infrastructure supports tests in two sorts of places, and they're
for two kinds of tests: you include **unit test**s inside of the crate itself,
and you place **integration test**s inside a `tests` directory. "Unit tests"
are small tests that test one focused unit, "integration tests" tests multiple
units in integration. That said, this is a social convention, they're no different
in syntax. Let's make a `tests` directory:

```{bash,ignore}
$ mkdir tests
```

Next, let's create an integration test in `tests/lib.rs`:

```{rust,no_run}
#[test]
fn foo() {
    assert!(false);
}
```

It doesn't matter what you name your test functions, though it's nice if
you give them descriptive names. You'll see why in a moment. We then use a
macro, `assert!`, to assert that something is true. In this case, we're giving
it `false`, so this test should fail. Let's try it!

```{notrust,ignore}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)
/home/you/projects/testing/src/main.rs:1:1: 3:2 warning: code is never used: `main`, #[warn(dead_code)] on by default
/home/you/projects/testing/src/main.rs:1 fn main() {
/home/you/projects/testing/src/main.rs:2     println!("Hello, world");
/home/you/projects/testing/src/main.rs:3 }

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test foo ... FAILED

failures:

---- foo stdout ----
        task 'foo' failed at 'assertion failed: false', /home/you/projects/testing/tests/lib.rs:3



failures:
    foo

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured

task '<main>' failed at 'Some tests failed', /home/you/src/rust/src/libtest/lib.rs:242
```

Lots of output! Let's break this down:

```{notrust,ignore}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)
```

You can run all of your tests with `cargo test`. This runs both your tests in
`tests`, as well as the tests you put inside of your crate.

```{notrust,ignore}
/home/you/projects/testing/src/main.rs:1:1: 3:2 warning: code is never used: `main`, #[warn(dead_code)] on by default
/home/you/projects/testing/src/main.rs:1 fn main() {
/home/you/projects/testing/src/main.rs:2     println!("Hello, world");
/home/you/projects/testing/src/main.rs:3 }
```

Rust has a **lint** called 'warn on dead code' used by default. A lint is a
bit of code that checks your code, and can tell you things about it. In this
case, Rust is warning us that we've written some code that's never used: our
`main` function. Of course, since we're running tests, we don't use `main`.
We'll turn this lint off for just this function soon. For now, just ignore this
output.

```{notrust,ignore}
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Wait a minute, zero tests? Didn't we define one? Yup. This output is from
attempting to run the tests in our crate, of which we don't have any.
You'll note that Rust reports on several kinds of tests: passed, failed,
ignored, and measured. The 'measured' tests refer to benchmark tests, which
we'll cover soon enough!

```{notrust,ignore}
running 1 test
test foo ... FAILED
```

Now we're getting somewhere. Remember when we talked about naming our tests
with good names? This is why. Here, it says 'test foo' because we called our
test 'foo.' If we had given it a good name, it'd be more clear which test
failed, especially as we accumulate more tests.

```{notrust,ignore}
failures:

---- foo stdout ----
        task 'foo' failed at 'assertion failed: false', /home/you/projects/testing/tests/lib.rs:3



failures:
    foo

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured

task '<main>' failed at 'Some tests failed', /home/you/src/rust/src/libtest/lib.rs:242
```

After all the tests run, Rust will show us any output from our failed tests.
In this instance, Rust tells us that our assertion failed, with false. This was
what we expected.

Whew! Let's fix our test:

```{rust}
#[test]
fn foo() {
    assert!(true);
}
```

And then try to run our tests again:

```{notrust,ignore}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)
/home/you/projects/testing/src/main.rs:1:1: 3:2 warning: code is never used: `main`, #[warn(dead_code)] on by default
/home/you/projects/testing/src/main.rs:1 fn main() {
/home/you/projects/testing/src/main.rs:2     println!("Hello, world");
/home/you/projects/testing/src/main.rs:3 }

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test foo ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
$
```

Nice! Our test passes, as we expected. Let's get rid of that warning for our `main`
function. Change your `src/main.rs` to look like this:

```{rust}
#[cfg(not(test))]
fn main() {
    println!("Hello, world");
}
```

This attribute combines two things: `cfg` and `not`. The `cfg` attribute allows
you to conditionally compile code based on something. The following item will
only be compiled if the configuration says it's true. And when Cargo compiles
our tests, it sets things up so that `cfg(test)` is true. But we want to only
include `main` when it's _not_ true. So we use `not` to negate things:
`cfg(not(test))` will only compile our code when the `cfg(test)` is false.

With this attribute, we won't get the warning:

```{notrust,ignore}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test foo ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

Nice. Okay, let's write a real test now. Change your `tests/lib.rs`
to look like this:

```{rust,ignore}
#[test]
fn math_checks_out() {
    let result = add_three_times_four(5i);

    assert_eq!(32i, result);
}
```

And try to run the test:

```{notrust,ignore}
$ cargo test
   Compiling testing v0.1.0 (file:/home/youg/projects/testing)
/home/youg/projects/testing/tests/lib.rs:3:18: 3:38 error: unresolved name `add_three_times_four`.
/home/youg/projects/testing/tests/lib.rs:3     let result = add_three_times_four(5i);
                                                            ^~~~~~~~~~~~~~~~~~~~
error: aborting due to previous error
Build failed, waiting for other jobs to finish...
Could not compile `testing`.

To learn more, run the command again with --verbose.
```

Rust can't find this function. That makes sense, as we didn't write it yet!

In order to share this codes with our tests, we'll need to make a library crate.
This is also just good software design: as we mentioned before, it's a good idea
to put most of your functionality into a library crate, and have your executable
crate use that library. This allows for code re-use.

To do that, we'll need to make a new module. Make a new file, `src/lib.rs`,
and put this in it:

```{rust}
fn add_three_times_four(x: int) -> int {
    (x + 3) * 4
}
```

We're calling this file `lib.rs` because it has the same name as our project,
and so it's named this, by convention.

We'll then need to use this crate in our `src/main.rs`:

```{rust,ignore}
extern crate testing;

#[cfg(not(test))]
fn main() {
    println!("Hello, world");
}
```

Finally, let's import this function in our `tests/lib.rs`:

```{rust,ignore}
extern crate testing;
use testing::add_three_times_four;

#[test]
fn math_checks_out() {
    let result = add_three_times_four(5i);

    assert_eq!(32i, result);
}
```

Let's give it a run:

```{ignore,notrust}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test math_checks_out ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

Great! One test passed. We've got an integration test showing that our public
method works, but maybe we want to test some of the internal logic as well.
While this function is simple, if it were more complicated, you can imagine
we'd need more tests. So let's break it up into two helper functions, and
write some unit tests to test those.

Change your `src/lib.rs` to look like this:

```{rust,ignore}
pub fn add_three_times_four(x: int) -> int {
    times_four(add_three(x))
}

fn add_three(x: int) -> int { x + 3 }

fn times_four(x: int) -> int { x * 4 }
```

If you run `cargo test`, you should get the same output:

```{ignore,notrust}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test math_checks_out ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

If we tried to write a test for these two new functions, it wouldn't
work. For example:

```{rust,ignore}
extern crate testing;
use testing::add_three_times_four;
use testing::add_three;

#[test]
fn math_checks_out() {
    let result = add_three_times_four(5i);

    assert_eq!(32i, result);
}

#[test]
fn test_add_three() {
    let result = add_three(5i);

    assert_eq!(8i, result);
}
```

We'd get this error:

```{notrust,ignore}
   Compiling testing v0.1.0 (file:/home/you/projects/testing)
/home/you/projects/testing/tests/lib.rs:3:5: 3:24 error: function `add_three` is private
/home/you/projects/testing/tests/lib.rs:3 use testing::add_three;
                                              ^~~~~~~~~~~~~~~~~~~
```

Right. It's private. So external, integration tests won't work. We need a
unit test. Open up your `src/lib.rs` and add this:

```{rust,ignore}
pub fn add_three_times_four(x: int) -> int {
    times_four(add_three(x))
}

fn add_three(x: int) -> int { x + 3 }

fn times_four(x: int) -> int { x * 4 }

#[cfg(test)]
mod test {
    use super::add_three;
    use super::times_four;

    #[test]
    fn test_add_three() {
        let result = add_three(5i);

        assert_eq!(8i, result);
    }

    #[test]
    fn test_times_four() {
        let result = times_four(5i);

        assert_eq!(20i, result);
    }
}
```

Let's give it a shot:

```{ignore,notrust}
$ cargo test
   Compiling testing v0.1.0 (file:/home/you/projects/testing)

running 1 test
test test::test_times_four ... ok
test test::test_add_three ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured


running 1 test
test math_checks_out ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

Cool! We now have two tests of our internal functions. You'll note that there
are three sets of output now: one for `src/main.rs`, one for `src/lib.rs`, and
one for `tests/lib.rs`. There's one interesting thing that we haven't talked
about yet, and that's these lines:

```{rust,ignore}
use super::add_three;
use super::times_four;
```

Because we've made a nested module, we can import functions from the parent
module by using `super`. Sub-modules are allowed to 'see' private functions in
the parent. We sometimes call this usage of `use` a 're-export,' because we're
exporting the name again, somewhere else.

We've now covered the basics of testing. Rust's tools are primitive, but they
work well in the simple cases. There are some Rustaceans working on building
more complicated frameworks on top of all of this, but thery're just starting
out.

# Pointers

In systems programming, pointers are an incredibly important topic. Rust has a
very rich set of pointers, and they operate differently than in many other
languages. They are important enough that we have a specific [Pointer
Guide](/guide-pointers.html) that goes into pointers in much detail. In fact,
while you're currently reading this guide, which covers the language in broad
overview, there are a number of other guides that put a specific topic under a
microscope. You can find the list of guides on the [documentation index
page](/index.html#guides).

In this section, we'll assume that you're familiar with pointers as a general
concept. If you aren't, please read the [introduction to
pointers](/guide-pointers.html#an-introduction) section of the Pointer Guide,
and then come back here. We'll wait.

Got the gist? Great. Let's talk about pointers in Rust.

## References

The most primitive form of pointer in Rust is called a **reference**.
References are created using the ampersand (`&`). Here's a simple
reference:

```{rust}
let x = 5i;
let y = &x;
```

`y` is a reference to `x`. To dereference (get the value being referred to
rather than the reference itself) `y`, we use the asterisk (`*`):

```{rust}
let x = 5i;
let y = &x;

assert_eq!(5i, *y);
```

Like any `let` binding, references are immutable by default.

You can declare that functions take a reference:

```{rust}
fn add_one(x: &int) -> int { *x + 1 }

fn main() {
    assert_eq!(6, add_one(&5));
}
```

As you can see, we can make a reference from a literal by applying `&` as well.
Of course, in this simple function, there's not a lot of reason to take `x` by
reference. It's just an example of the syntax.

Because references are immutable, you can have multiple references that
**alias** (point to the same place):

```{rust}
let x = 5i;
let y = &x;
let z = &x;
```

We can make a mutable reference by using `&mut` instead of `&`:

```{rust}
let mut x = 5i;
let y = &mut x;
```

Note that `x` must also be mutable. If it isn't, like this:

```{rust,ignore}
let x = 5i;
let y = &mut x;
```

Rust will complain:

```{ignore,notrust}
6:19 error: cannot borrow immutable local variable `x` as mutable
 let y = &mut x;
              ^
```

We don't want a mutable reference to immutable data! This error message uses a
term we haven't talked about yet, 'borrow.' We'll get to that in just a moment.

This simple example actually illustrates a lot of Rust's power: Rust has
prevented us, at compile time, from breaking our own rules. Because Rust's
references check these kinds of rules entirely at compile time, there's no
runtime overhead for this safety.  At runtime, these are the same as a raw
machine pointer, like in C or C++.  We've just double-checked ahead of time
that we haven't done anything dangerous.

Rust will also prevent us from creating two mutable references that alias.
This won't work:

```{rust,ignore}
let mut x = 5i;
let y = &mut x;
let z = &mut x;
```

It gives us this error:

```{notrust,ignore}
error: cannot borrow `x` as mutable more than once at a time
     let z = &mut x;
                  ^
note: previous borrow of `x` occurs here; the mutable borrow prevents subsequent moves, borrows, or modification of `x` until the borrow ends
     let y = &mut x;
                  ^
note: previous borrow ends here
 fn main() {
     let mut x = 5i;
     let y = &mut x;
     let z = &mut x;
 }
 ^
```

This is a big error message. Let's dig into it for a moment. There are three
parts: the error and two notes. The error says what we expected, we cannot have
two pointers that point to the same memory.

The two notes give some extra context. Rust's error messages often contain this
kind of extra information when the error is complex. Rust is telling us two
things: first, that the reason we cannot **borrow** `x` as `z` is that we
previously borrowed `x` as `y`. The second note shows where `y`'s borrowing
ends.

Wait, borrowing?

In order to truly understand this error, we have to learn a few new concepts:
**ownership**, **borrowing**, and **lifetimes**.

## Ownership, borrowing, and lifetimes

## Boxes

All of our references so far have been to variables we've created on the stack.
In Rust, the simplest way to allocate heap variables is using a *box*.  To
create a box, use the `box` keyword:
 
```{rust}
let x = box 5i;
```

This allocates an integer `5` on the heap, and creates a binding `x` that
refers to it.. The great thing about boxed pointers is that we don't have to
manually free this allocation! If we write 

```{rust}
{
    let x = box 5i;
    // do stuff
}
```

then Rust will automatically free `x` at the end of the block. This isn't
because Rust has a garbage collector -- it doesn't. Instead, Rust uses static
analysis to determine the *lifetime* of `x`, and then generates code to free it
once it's sure the `x` won't be used again. This Rust code will do the same
thing as the following C code:

```{c,ignore}
{
    int *x = (int *)malloc(sizeof(int));
    // do stuff
    free(x);
}
```

This means we get the benefits of manual memory management, but the compiler
ensures that we don't do something wrong. We can't forget to `free` our memory.

Boxes are the sole owner of their contents, so you cannot take a mutable
reference to them and then use the original box:

```{rust,ignore}
let mut x = box 5i;
let y = &mut x;

*x; // you might expect 5, but this is actually an error
```

This gives us this error:

```{notrust,ignore}
8:7 error: cannot use `*x` because it was mutably borrowed
 *x;
 ^~
 6:19 note: borrow of `x` occurs here
 let y = &mut x;
              ^
```

As long as `y` is borrowing the contents, we cannot use `x`. After `y` is
done borrowing the value, we can use it again. This works fine:

```{rust}
let mut x = box 5i;

{
    let y = &mut x;
} // y goes out of scope at the end of the block

*x;
```

## Rc and Arc

Sometimes, you need to allocate something on the heap, but give out multiple
references to the memory. Rust's `Rc<T>` (pronounced 'arr cee tee') and
`Arc<T>` types (again, the `T` is for generics, we'll learn more later) provide
you with this ability.  **Rc** stands for 'reference counted,' and **Arc** for
'atomically reference counted.' This is how Rust keeps track of the multiple
owners: every time we make a new reference to the `Rc<T>`, we add one to its
internal 'reference count.' Every time a reference goes out of scope, we
subtract one from the count. When the count is zero, the `Rc<T>` can be safely
deallocated. `Arc<T>` is almost identical to `Rc<T>`, except for one thing: The
'atomically' in 'Arc' means that increasing and decreasing the count uses a
thread-safe mechanism to do so. Why two types? `Rc<T>` is faster, so if you're
not in a multi-threaded scenario, you can have that advantage. Since we haven't
talked about threading yet in Rust, we'll show you `Rc<T>` for the rest of this
section.

To create an `Rc<T>`, use `Rc::new()`:

```{rust}
use std::rc::Rc;

let x = Rc::new(5i);
```

To create a second reference, use the `.clone()` method:

```{rust}
use std::rc::Rc;

let x = Rc::new(5i);
let y = x.clone();
```

The `Rc<T>` will live as long as any of its references are alive. After they
all go out of scope, the memory will be `free`d.

If you use `Rc<T>` or `Arc<T>`, you have to be careful about introducing
cycles. If you have two `Rc<T>`s that point to each other, the reference counts
will never drop to zero, and you'll have a memory leak. To learn more, check
out [the section on `Rc<T>` and `Arc<T>` in the pointers
guide](http://doc.rust-lang.org/guide-pointers.html#rc-and-arc).

# Patterns

# Lambdas

# iterators

# Generics

# Traits

# Operators and built-in Traits

# Tasks

# Macros

# Unsafe

