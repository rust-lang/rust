% Rust Language Tutorial

# Introduction

## Scope

This is a tutorial for the Rust programming language. It assumes the
reader is familiar with the basic concepts of programming, and has
programmed in one or more other languages before. It will often make
comparisons to other languages in the C family. The tutorial covers
the whole language, though not with the depth and precision of the
[language reference](rust.html).

## Language overview

Rust is a systems programming language with a focus on type safety,
memory safety, concurrency and performance. It is intended for writing
large, high performance applications while preventing several classes
of errors commonly found in languages like C++. Rust has a
sophisticated memory model that enables many of the efficient data
structures used in C++ while disallowing invalid memory access that
would otherwise cause segmentation faults. Like other systems
languages it is statically typed and compiled ahead of time.

As a multi-paradigm language it has strong support for writing code in
procedural, functional and object-oriented styles. Some of it's nice
high-level features include:

* Pattern matching and algebraic data types (enums) - common in functional
  languages, pattern matching on ADTs provides a compact and expressive
  way to encode program logic
* Task-based concurrency - Rust uses lightweight tasks that do not share
  memory
* Higher-order functions - Closures in Rust are very powerful and used
  pervasively
* Polymorphism - Rust's type system features a unique combination of
  Java-style interfaces and Haskell-style typeclasses
* Generics - Functions and types can be parameterized over generic
  types with optional type constraints

## First impressions

As a curly-brace language in the tradition of C, C++, and JavaScript,
Rust looks a lot like other languages you may be familiar with.

~~~~
fn boring_old_factorial(n: int) -> int {
    let mut result = 1, i = 1;
    while i <= n {
        result *= i;
        i += 1;
    }
    ret result;
}
~~~~

Several differences from C stand out. Types do not come before, but
after variable names (preceded by a colon). For local variables
(introduced with `let`), types are optional, and will be inferred when
left off. Constructs like `while` and `if` do not require parentheses
around the condition (though they allow them). Also, there's a
tendency towards aggressive abbreviation in the keywords—`fn` for
function, `ret` for return.

You should, however, not conclude that Rust is simply an evolution of
C. As will become clear in the rest of this tutorial, it goes in quite
a different direction, with efficient, strongly-typed and memory-safe
support for many high-level idioms.

Here's a parallel game of rock, paper, scissors to whet your appetite.

~~~~
use std;

import comm::{listen, methods};
import task::spawn;
import iter::repeat;
import rand::{seeded_rng, seed};
import uint::range;
import io::println;

fn main() {
    // Open a channel to receive game results
    do listen |result_from_game| {

        let times = 10;
        let player1 = "graydon";
        let player2 = "patrick";

        for repeat(times) {
            // Start another task to play the game
            do spawn |copy player1, copy player2| {
                let outcome = play_game(player1, player2);
                result_from_game.send(outcome);
            }
        }

        // Report the results as the games complete
        for range(0, times) |round| {
            let winner = result_from_game.recv();
            println(#fmt("%s wins round #%u", winner, round));
        }
    }

    fn play_game(player1: str, player2: str) -> str {

        // Our rock/paper/scissors types
        enum gesture {
            rock, paper, scissors
        }

        let rng = seeded_rng(seed());
        // A small inline function for picking an RPS gesture
        let pick = || [rock, paper, scissors][rng.gen_uint() % 3];

        // Pick two gestures and decide the result
        alt (pick(), pick()) {
            (rock, scissors) | (paper, rock) | (scissors, paper) { copy player1 }
            (scissors, rock) | (rock, paper) | (paper, scissors) { copy player2 }
            _ { "tie" }
        }
    }
}
~~~~

## Conventions

Throughout the tutorial, words that indicate language keywords or
identifiers defined in the example code are displayed in `code font`.

Code snippets are indented, and also shown in a monospaced font. Not
all snippets constitute whole programs. For brevity, we'll often show
fragments of programs that don't compile on their own. To try them
out, you might have to wrap them in `fn main() { ... }`, and make sure
they don't contain references to things that aren't actually defined.

> ***Warning:*** Rust is a language under heavy development. Notes
> about potential changes to the language, implementation
> deficiencies, and other caveats appear offset in blockquotes.

## Disclaimer

Rust is a language under development. The general flavor of the
language has settled, but details will continue to change as it is
further refined. Nothing in this tutorial is final, and though we try
to keep it updated, it is possible that the text occasionally does not
reflect the actual state of the language.

# Getting started

## Installation

The Rust compiler currently must be built from a [tarball][]. We hope
to be distributing binary packages for various operating systems in
the future.

The Rust compiler is slightly unusual in that it is written in Rust
and therefore must be built by a precompiled "snapshot" version of
itself (made in an earlier state of development). As such, source
builds require that:

  * You are connected to the internet, to fetch snapshots.
  * You can at least execute snapshot binaries of one of the forms we
    offer them in. Currently we build and test snapshots on:
    * Windows (7, server 2008 r2) x86 only
    * Linux (various distributions) x86 and x86-64
    * OSX 10.6 ("Snow Leopard") or 10.7 ("Lion") x86 and x86-64

You may find other platforms work, but these are our "tier 1" supported
build environments that are most likely to work. Further platforms will
be added to the list in the future via cross-compilation.

To build from source you will also need the following prerequisite
packages:

  * g++ 4.4 or clang++ 3.x
  * python 2.6 or later
  * perl 5.0 or later
  * gnu make 3.81 or later
  * curl

Assuming you're on a relatively modern *nix system and have met the
prerequisites, something along these lines should work. Building from
source on Windows requires some extra steps: please see the [getting
started][wiki-get-started] page on the Rust wiki.

~~~~ {.notrust}
$ wget http://dl.rust-lang.org/dist/rust-0.2.tar.gz
$ tar -xzf rust-0.2.tar.gz
$ cd rust-0.2
$ ./configure
$ make && make install
~~~~

You may need to use `sudo make install` if you do not normally have
permission to modify the destination directory. The install locations
can be adjusted by passing a `--prefix` argument to
`configure`. Various other options are also supported, pass `--help`
for more information on them.

When complete, `make install` will place the following programs into
`/usr/local/bin`:

  * `rustc`, the Rust compiler
  * `rustdoc`, the API-documentation tool 
  * `cargo`, the Rust package manager

[wiki-get-started]: https://github.com/mozilla/rust/wiki/Note-getting-started-developing-Rust
[tarball]: http://dl.rust-lang.org/dist/rust-0.2.tar.gz

## Compiling your first program

Rust program files are, by convention, given the extension `.rs`. Say
we have a file `hello.rs` containing this program:

~~~~
fn main(args: ~[str]) {
    io::println("hello world from '" + args[0] + "'!");
}
~~~~

If the Rust compiler was installed successfully, running `rustc
hello.rs` will produce a binary called `hello` (or `hello.exe`).

If you modify the program to make it invalid (for example, by changing
 `io::println` to some nonexistent function), and then compile it,
 you'll see an error message like this:

~~~~ {.notrust}
hello.rs:2:4: 2:16 error: unresolved name: io::print_it
hello.rs:2     io::print_it("hello world from '" + args[0] + "'!");
               ^~~~~~~~~~~~
~~~~

The Rust compiler tries to provide useful information when it runs
into an error.

## Anatomy of a Rust program

In its simplest form, a Rust program is simply a `.rs` file with some
types and functions defined in it. If it has a `main` function, it can
be compiled to an executable. Rust does not allow code that's not a
declaration to appear at the top level of the file—all statements must
live inside a function.

Rust programs can also be compiled as libraries, and included in other
programs. The `use std` directive that appears at the top of a lot of
examples imports the [standard library][std]. This is described in more
detail [later on](#modules-and-crates).

[std]: http://doc.rust-lang.org/doc/std

## Editing Rust code

There are Vim highlighting and indentation scripts in the Rust source
distribution under `src/etc/vim/`, and an emacs mode under
`src/etc/emacs/`.

Other editors are not provided for yet. If you end up writing a Rust
mode for your favorite editor, let us know so that we can link to it.

# Syntax Basics

## Braces

Assuming you've programmed in any C-family language (C++, Java,
JavaScript, C#, or PHP), Rust will feel familiar. The main surface
difference to be aware of is that the bodies of `if` statements and of
`while` loops *have* to be wrapped in brackets. Single-statement,
bracket-less bodies are not allowed.

If the verbosity of that bothers you, consider the fact that this
allows you to omit the parentheses around the condition in `if`,
`while`, and similar constructs. This will save you two characters
every time. As a bonus, you no longer have to spend any mental energy
on deciding whether you need to add braces or not, or on adding them
after the fact when adding a statement to an `if` branch.

Accounting for these differences, the surface syntax of Rust
statements and expressions is C-like. Function calls are written
`myfunc(arg1, arg2)`, operators have mostly the same name and
precedence that they have in C, comments look the same, and constructs
like `if` and `while` are available:

~~~~
# fn call_a_function(_a: int) {}
fn main() {
    if 1 < 2 {
        while false { call_a_function(10 * 4); }
    } else if 4 < 3 || 3 < 4 {
        // Comments are C++-style too
    } else {
        /* Multi-line comment syntax */
    }
}
~~~~

## Expression syntax

Though it isn't apparent in all code, there is a fundamental
difference between Rust's syntax and the predecessors in this family
of languages. A lot of things that are statements in C are expressions
in Rust. This allows for useless things like this (which passes
nil—the void type—to a function):

~~~~
# fn a_function(_a: ()) {}
a_function(while false {});
~~~~

But also useful things like this:

~~~~
# fn the_stars_align() -> bool { false }
# fn something_else() -> bool { true }
let x = if the_stars_align() { 4 }
        else if something_else() { 3 }
        else { 0 };
~~~~

This piece of code will bind the variable `x` to a value depending on
the conditions. Note the condition bodies, which look like `{
expression }`. The lack of a semicolon after the last statement in a
braced block gives the whole block the value of that last expression.
If the branches of the `if` had looked like `{ 4; }`, the above
example would simply assign nil (void) to `x`. But without the
semicolon, each branch has a different value, and `x` gets the value
of the branch that was taken.

This also works for function bodies. This function returns a boolean:

~~~~
fn is_four(x: int) -> bool { x == 4 }
~~~~

In short, everything that's not a declaration (`let` for variables,
`fn` for functions, et cetera) is an expression.

If all those things are expressions, you might conclude that you have
to add a terminating semicolon after *every* statement, even ones that
are not traditionally terminated with a semicolon in C (like `while`).
That is not the case, though. Expressions that end in a block only
need a semicolon if that block contains a trailing expression. `while`
loops do not allow trailing expressions, and `if` statements tend to
only have a trailing expression when you want to use their value for
something—in which case you'll have embedded it in a bigger statement,
like the `let x = ...` example above.

## Identifiers

Rust identifiers must start with an alphabetic character or an
underscore, and after that may contain any alphanumeric character, and
more underscores.

The double-colon (`::`) is used as a module separator, so
`io::println` means 'the thing named `println` in the module
named `io`'.

Rust will normally emit warnings about unused variables. These can be
suppressed by using a variable name that starts with an underscore.

~~~~
fn this_warns(x: int) {}
fn this_doesnt(_x: int) {}
~~~~

## Variable declaration

The `let` keyword, as we've seen, introduces a local variable. Local
variables are immutable by default: `let mut` can be used to introduce
a local variable that can be reassigned.  Global constants can be
defined with `const`:

~~~~
use std;
const repeat: uint = 5u;
fn main() {
    let hi = "Hi!";
    let mut count = 0u;
    while count < repeat {
        io::println(hi);
        count += 1u;
    }
}
~~~~

Local variables may shadow earlier declarations, causing the
previous variable to go out of scope.

~~~~
let my_favorite_value: float = 57.8;
let my_favorite_value: int = my_favorite_value as int;
~~~~

## Types

The `-> bool` in the `is_four` example is the way a function's return
type is written. For functions that do not return a meaningful value
(these conceptually return nil in Rust), you can optionally say `->
()` (`()` is how nil is written), but usually the return annotation is
simply left off, as in the `fn main() { ... }` examples we've seen
earlier.

Every argument to a function must have its type declared (for example,
`x: int`). Inside the function, type inference will be able to
automatically deduce the type of most locals (generic functions, which
we'll come back to later, will occasionally need additional
annotation). Locals can be written either with or without a type
annotation:

~~~~
// The type of this vector will be inferred based on its use.
let x = ~[];
# vec::map(x, fn&(&&_y:int) -> int { _y });
// Explicitly say this is a vector of integers.
let y: ~[int] = ~[];
~~~~

The basic types are written like this:

`()`
  : Nil, the type that has only a single value.

`bool`
  : Boolean type, with values `true` and `false`.

`int`
  : A machine-pointer-sized integer.

`uint`
  : A machine-pointer-sized unsigned integer.

`i8`, `i16`, `i32`, `i64`
  : Signed integers with a specific size (in bits).

`u8`, `u16`, `u32`, `u64`
  : Unsigned integers with a specific size.

`f32`, `f64`
  : Floating-point types.

`float`
  : The largest floating-point type efficiently supported on the target machine.

`char`
  : A character is a 32-bit Unicode code point.

`str`
  : String type. A string contains a UTF-8 encoded sequence of characters.

These can be combined in composite types, which will be described in
more detail later on (the `T`s here stand for any other type):

`~[T]`
  : Vector type.

`~[mut T]`
  : Mutable vector type.

`(T1, T2)`
  : Tuple type. Any arity above 1 is supported.

`{field1: T1, field2: T2}`
  : Record type.

`fn(arg1: T1, arg2: T2) -> T3`, `fn@()`, `fn~()`, `fn&()`
  : Function types.

`@T`, `~T`, `*T`
  : Pointer types.

Types can be given names with `type` declarations:

~~~~
type monster_size = uint;
~~~~

This will provide a synonym, `monster_size`, for unsigned integers. It
will not actually create a new type—`monster_size` and `uint` can be
used interchangeably, and using one where the other is expected is not
a type error. Read about [single-variant enums](#single_variant_enum)
further on if you need to create a type name that's not just a
synonym.

## Numeric literals

Integers can be written in decimal (`144`), hexadecimal (`0x90`), and
binary (`0b10010000`) base.

If you write an integer literal without a suffix (`3`, `-500`, etc.),
the Rust compiler will try to infer its type based on type annotations
and function signatures in the surrounding program.  For example, here
the type of `x` is inferred to be `u16` because it is passed to a
function that takes a `u16` argument:

~~~~~
let x = 3;

fn identity_u16(n: u16) -> u16 { n }

identity_u16(x);
~~~~

On the other hand, if the program gives conflicting information about
what the type of the unsuffixed literal should be, you'll get an error
message.

~~~~~{.xfail-test}
let x = 3;
let y: i32 = 3;

fn identity_u8(n: u8) -> u8 { n }
fn identity_u16(n: u16) -> u16 { n }

identity_u8(x);  // after this, `x` is assumed to have type `u8`
identity_u16(x); // raises a type error (expected `u16` but found `u8`)
identity_u16(y); // raises a type error (expected `u16` but found `i32`)
~~~~

In the absence of any type annotations at all, Rust will assume that
an unsuffixed integer literal has type `int`.

~~~~
let n = 50;
log(error, n); // n is an int
~~~~

It's also possible to avoid any type ambiguity by writing integer
literals with a suffix.  The suffixes `i` and `u` are for the types
`int` and `uint`, respectively: the literal `-3i` has type `int`,
while `127u` has type `uint`.  For the fixed-size integer types, just
suffix the literal with the type name: `255u8`, `50i64`, etc.

Note that, in Rust, no implicit conversion between integer types
happens. If you are adding one to a variable of type `uint`, saying
`+= 1u8` will give you a type error.

Floating point numbers are written `0.0`, `1e6`, or `2.1e-4`. Without
a suffix, the literal is assumed to be of type `float`. Suffixes `f32`
and `f64` can be used to create literals of a specific type. The
suffix `f` can be used to write `float` literals without a dot or
exponent: `3f`.

## Other literals

The nil literal is written just like the type: `()`. The keywords
`true` and `false` produce the boolean literals.

Character literals are written between single quotes, as in `'x'`. You
may put non-ascii characters between single quotes (your source files
should be encoded as UTF-8). Rust understands a number of
character escapes, using the backslash character:

`\n`
  : A newline (Unicode character 10).

`\r`
  : A carriage return (13).

`\t`
  : A tab character (9).

`\\`, `\'`, `\"`
  : Simply escapes the following character.

`\xHH`, `\uHHHH`, `\UHHHHHHHH`
  : Unicode escapes, where the `H` characters are the hexadecimal digits that
    form the character code.

String literals allow the same escape sequences. They are written
between double quotes (`"hello"`). Rust strings may contain newlines.
When a newline is preceded by a backslash, it, and all white space
following it, will not appear in the resulting string literal. So
this is equivalent to `"abc"`:

~~~~
let s = "a\
         b\
         c";
~~~~

## Operators

Rust's set of operators contains very few surprises. Binary arithmetic
is done with `*`, `/`, `%`, `+`, and `-` (multiply, divide, remainder,
plus, minus). `-` is also a unary prefix operator that does negation.

Binary shifting is done with `>>` (shift right), and `<<` (shift
left). Shift right is arithmetic if the value is signed and logical if
the value is unsigned. Logical bitwise operators are `&`, `|`, and `^`
(and, or, and exclusive or), and unary `!` for bitwise negation (or
boolean negation when applied to a boolean value).

The comparison operators are the traditional `==`, `!=`, `<`, `>`,
`<=`, and `>=`. Short-circuiting (lazy) boolean operators are written
`&&` (and) and `||` (or).

For type casting, Rust uses the binary `as` operator, which has high
precedence, just lower than multiplication and division.  It takes an
expression on the left side, and a type on the right side, and will,
if a meaningful conversion exists, convert the result of the
expression to the given type.

~~~~
let x: float = 4.0;
let y: uint = x as uint;
assert y == 4u;
~~~~

The main difference with C is that `++` and `--` are missing, and that
the logical bitwise operators have higher precedence — in C, `x & 2 > 0`
comes out as `x & (2 > 0)`, in Rust, it means `(x & 2) > 0`, which is
more likely to be what you expect (unless you are a C veteran).

## Attributes

Every definition can be annotated with attributes. Attributes are meta
information that can serve a variety of purposes. One of those is
conditional compilation:

~~~~
#[cfg(windows)]
fn register_win_service() { /* ... */ }
~~~~

This will cause the function to vanish without a trace during
compilation on a non-Windows platform, much like `#ifdef` in C.

Attributes are always wrapped in hash-braces (`#[attr]`). Inside the
braces, a small minilanguage is supported, whose interpretation
depends on the attribute that's being used. The simplest form is a
plain name (as in `#[test]`, which is used by the [built-in test
framework](#testing)). A name-value pair can be provided using an `=`
character followed by a literal (as in `#[license = "BSD"]`, which is
a valid way to annotate a Rust program as being released under a
BSD-style license). Finally, you can have a name followed by a
comma-separated list of nested attributes, as in this
[crate](#modules-and-crates) metadata declaration:

~~~~ {.ignore}
#[link(name = "std",
       vers = "0.1",
       url = "http://rust-lang.org/src/std")];
~~~~

An attribute without a semicolon following it applies to the
definition that follows it. When terminated with a semicolon, it
applies to the module or crate in which it appears.

## Syntax extensions

There are plans to support user-defined syntax (macros) in Rust. This
currently only exists in very limited form.

The compiler defines a few built-in syntax extensions. The most useful
one is `#fmt`, a printf-style text formatting macro that is expanded
at compile time.

~~~~
io::println(#fmt("%s is %d", "the answer", 42));
~~~~

`#fmt` supports most of the directives that [printf][pf] supports, but
will give you a compile-time error when the types of the directives
don't match the types of the arguments.

[pf]: http://en.cppreference.com/w/cpp/io/c/fprintf

All syntax extensions look like `#word`. Another built-in one is
`#env`, which will look up its argument as an environment variable at
compile-time.

~~~~
io::println(#env("PATH"));
~~~~

# Control structures

## Conditionals

We've seen `if` pass by a few times already. To recap, braces are
compulsory, an optional `else` clause can be appended, and multiple
`if`/`else` constructs can be chained together:

~~~~
if false {
    io::println("that's odd");
} else if true {
    io::println("right");
} else {
    io::println("neither true nor false");
}
~~~~

The condition given to an `if` construct *must* be of type boolean (no
implicit conversion happens). If the arms return a value, this value
must be of the same type for every arm in which control reaches the
end of the block:

~~~~
fn signum(x: int) -> int {
    if x < 0 { -1 }
    else if x > 0 { 1 }
    else { ret 0; }
}
~~~~

The `ret` (return) and its semicolon could have been left out without
changing the meaning of this function, but it illustrates that you
will not get a type error in this case, although the last arm doesn't
have type `int`, because control doesn't reach the end of that arm
(`ret` is jumping out of the function).

## Pattern matching

Rust's `alt` construct is a generalized, cleaned-up version of C's
`switch` construct. You provide it with a value and a number of arms,
each labelled with a pattern, and it will execute the arm that matches
the value.

~~~~
# let my_number = 1;
alt my_number {
  0       { io::println("zero"); }
  1 | 2   { io::println("one or two"); }
  3 to 10 { io::println("three to ten"); }
  _       { io::println("something else"); }
}
~~~~

There is no 'falling through' between arms, as in C—only one arm is
executed, and it doesn't have to explicitly `break` out of the
construct when it is finished.

The part to the left of each arm is called the pattern. Literals are
valid patterns, and will match only their own value. The pipe operator
(`|`) can be used to assign multiple patterns to a single arm. Ranges
of numeric literal patterns can be expressed with `to`. The underscore
(`_`) is a wildcard pattern that matches everything.

If the arm with the wildcard pattern was left off in the above
example, running it on a number greater than ten (or negative) would
cause a run-time failure. When no arm matches, `alt` constructs do not
silently fall through—they blow up instead.

A powerful application of pattern matching is *destructuring*, where
you use the matching to get at the contents of data types. Remember
that `(float, float)` is a tuple of two floats:

~~~~
fn angle(vec: (float, float)) -> float {
    alt vec {
      (0f, y) if y < 0f { 1.5 * float::consts::pi }
      (0f, y) { 0.5 * float::consts::pi }
      (x, y) { float::atan(y / x) }
    }
}
~~~~

A variable name in a pattern matches everything, *and* binds that name
to the value of the matched thing inside of the arm block. Thus, `(0f,
y)` matches any tuple whose first element is zero, and binds `y` to
the second element. `(x, y)` matches any tuple, and binds both
elements to a variable.

Any `alt` arm can have a guard clause (written `if EXPR`), which is
an expression of type `bool` that determines, after the pattern is
found to match, whether the arm is taken or not. The variables bound
by the pattern are available in this guard expression.

## Destructuring let

To a limited extent, it is possible to use destructuring patterns when
declaring a variable with `let`. For example, you can say this to
extract the fields from a tuple:

~~~~
# fn get_tuple_of_two_ints() -> (int, int) { (1, 1) }
let (a, b) = get_tuple_of_two_ints();
~~~~

This will introduce two new variables, `a` and `b`, bound to the
content of the tuple.

You may only use irrefutable patterns—patterns that can never fail to
match—in let bindings. Other types of patterns, such as literals, are
not allowed.

## Loops

`while` produces a loop that runs as long as its given condition
(which must have type `bool`) evaluates to true. Inside a loop, the
keyword `break` can be used to abort the loop, and `again` can be used
to abort the current iteration and continue with the next.

~~~~
let mut cake_amount = 8;
while cake_amount > 0 {
    cake_amount -= 1;
}
~~~~

`loop` is the preferred way of writing `while true`:

~~~~
let mut x = 5;
loop {
    x += x - 3;
    if x % 5 == 0 { break; }
    io::println(int::str(x));
}
~~~~

This code prints out a weird sequence of numbers and stops as soon as
it finds one that can be divided by five.

For more involved iteration, such as going over the elements of a
collection, Rust uses higher-order functions. We'll come back to those
in a moment.

## Failure

The `fail` keyword causes the current [task](#tasks) to fail. You use
it to indicate unexpected failure, much like you'd use `abort` in a
C program or a fatal exception in a C++ program.

There is no way for the current task to resume execution after
failure; failure is nonrecoverable. It is, however, possible for
*another* task to handle the failure, allowing the program to continue
running.

`fail` takes an optional argument specifying the reason for the
failure. It must have type `str`.

In addition to the `fail` statement, the following circumstances cause
task failure:

* Accessing an out-of-bounds element of a vector.

* Having no clauses match when evaluating an `alt check` expression.

* An assertion failure.

* Integer division by zero.

* Running out of memory.

## Assertions

The keyword `assert`, followed by an expression with boolean type,
will check that the given expression results in `true`, and cause a
failure otherwise. It is typically used to double-check things that
*should* hold at a certain point in a program. `assert` statements are
always active; there is no way to build Rust code with assertions
disabled.

~~~~
let mut x = 100;
while (x > 10) { x -= 10; }
assert x == 10;
~~~~

## Logging

Rust has a built-in logging mechanism, using the `log` statement.
Logging is polymorphic—any type of value can be logged, and the
runtime will do its best to output a textual representation of the
value.

~~~~
log(warn, "hi");
log(error, (1, ~[2.5, -1.8]));
~~~~

The first argument is the log level (levels `debug`, `info`, `warn`,
and `error` are predefined), and the second is the value to log. By
default, you *will not* see the output of that first log statement,
which has `warn` level. The environment variable `RUST_LOG` controls
which log level is used. It can contain a comma-separated list of
paths for modules that should be logged. For example, running `rustc`
with `RUST_LOG=rustc::front::attr` will turn on logging in its
attribute parser. If you compile a program named `foo.rs`, its
top-level module will be called `foo`, and you can set `RUST_LOG` to
`foo` to enable `warn`, `info` and `debug` logging for the module.

Turned-off `log` statements impose minimal overhead on the code that
contains them, because the arguments to `log` are evaluated lazily.
So except in code that needs to be really, really fast,
you should feel free to scatter around debug logging statements, and
leave them in.

Three macros that combine text-formatting (as with `#fmt`) and logging
are available. These take a string and any number of format arguments,
and will log the formatted string:

~~~~
# fn get_error_string() -> str { "boo" }
#warn("only %d seconds remaining", 10);
#error("fatal: %s", get_error_string());
~~~~

Because the macros `#debug`, `#warn`, and `#error` expand to calls to `log`,
their arguments are also lazily evaluated.

# Functions

Like all other static declarations, such as `type`, functions can be
declared both at the top level and inside other functions (or modules,
which we'll come back to [later](#modules-and-crates)).

We've already seen several function definitions. They are introduced
with the `fn` keyword, the type of arguments are specified following
colons and the return type follows the arrow.

~~~~
fn int_to_str(i: int) -> str {
    ret "tube sock";
}
~~~~

The `ret` keyword immediately returns from the body of a function. It
is optionally followed by an expression to return. A function can
also return a value by having its top level block produce an
expression.

~~~~
# const copernicus: int = 0;
fn int_to_str(i: int) -> str {
    if i == copernicus {
        ret "tube sock";
    } else {
        ret "violin";
    }
}
~~~~

~~~~
# const copernicus: int = 0;
fn int_to_str(i: int) -> str {
    if i == copernicus { "tube sock" }
    else { "violin" }
}
~~~~

Functions that do not return a value are said to return nil, `()`,
and both the return type and the return value may be omitted from
the definition. The following two functions are equivalent.

~~~~
fn do_nothing_the_hard_way() -> () { ret (); }

fn do_nothing_the_easy_way() { }
~~~~

Some functions (such as the C function `exit`) never return normally.
In Rust, these are annotated with the pseudo-return type '`!`':

~~~~
fn dead_end() -> ! { fail }
~~~~

This helps the compiler avoid spurious error messages. For example,
the following code would be a type error if `dead_end` would be
expected to return.

~~~~
# fn can_go_left() -> bool { true }
# fn can_go_right() -> bool { true }
# enum dir { left, right }
# fn dead_end() -> ! { fail; }
let dir = if can_go_left() { left }
          else if can_go_right() { right }
          else { dead_end(); };
~~~~

# Basic datatypes

The core datatypes of Rust are structural records, enums (tagged
unions, algebraic data types), and tuples. They are immutable
by default.

~~~~
type point = {x: float, y: float};

enum shape {
    circle(point, float),
    rectangle(point, point)
}
~~~~

## Records

Rust record types are written `{field1: T1, field2: T2 [, ...]}`,
where `T1`, `T2`, ... denote types.  Record literals are written in
the same way, but with expressions instead of types. They are quite
similar to C structs, and even laid out the same way in memory (so you
can read from a Rust struct in C, and vice-versa). The dot operator is
used to access record fields (`mypoint.x`).

Fields that you want to mutate must be explicitly marked `mut`.

~~~~
type stack = {content: ~[int], mut head: uint};
~~~~

With such a type, you can do `mystack.head += 1u`. If `mut` were
omitted from the type, such an assignment would result in a type
error.

To create a new record based on the value of an existing record
you construct it using the `with` keyword:

~~~~
let oldpoint = {x: 10f, y: 20f};
let newpoint = {x: 0f with oldpoint};
assert newpoint == {x: 0f, y: 20f};
~~~~

This will create a new record, copying all the fields from `oldpoint`
into it, except for the ones that are explicitly set in the literal.

Rust record types are *structural*. This means that `{x: float, y:
float}` is not just a way to define a new type, but is the actual name
of the type. Record types can be used without first defining them. If
module A defines `type point = {x: float, y: float}`, and module B,
without knowing anything about A, defines a function that returns an
`{x: float, y: float}`, you can use that return value as a `point` in
module A. (Remember that `type` defines an additional name for a type,
not an actual new type.)

## Record patterns

Records can be destructured in `alt` patterns. The basic syntax is
`{fieldname: pattern, ...}`, but the pattern for a field can be
omitted as a shorthand for simply binding the variable with the same
name as the field.

~~~~
# let mypoint = {x: 0f, y: 0f};
alt mypoint {
    {x: 0f, y: y_name} { /* Provide sub-patterns for fields */ }
    {x, y}             { /* Simply bind the fields */ }
}
~~~~

The field names of a record do not have to appear in a pattern in the
same order they appear in the type. When you are not interested in all
the fields of a record, a record pattern may end with `, _` (as in
`{field1, _}`) to indicate that you're ignoring all other fields.

## Enums

Enums are datatypes that have several alternate representations. For
example, consider the type shown earlier:

~~~~
# type point = {x: float, y: float};
enum shape {
    circle(point, float),
    rectangle(point, point)
}
~~~~

A value of this type is either a circle, in which case it contains a
point record and a float, or a rectangle, in which case it contains
two point records. The run-time representation of such a value
includes an identifier of the actual form that it holds, much like the
'tagged union' pattern in C, but with better ergonomics.

The above declaration will define a type `shape` that can be used to
refer to such shapes, and two functions, `circle` and `rectangle`,
which can be used to construct values of the type (taking arguments of
the specified types). So `circle({x: 0f, y: 0f}, 10f)` is the way to
create a new circle.

Enum variants need not have type parameters. This, for example, is
equivalent to a C enum:

~~~~
enum direction {
    north,
    east,
    south,
    west
}
~~~~

This will define `north`, `east`, `south`, and `west` as constants,
all of which have type `direction`.

When an enum is C-like, that is, when none of the variants have
parameters, it is possible to explicitly set the discriminator values
to an integer value:

~~~~
enum color {
  red = 0xff0000,
  green = 0x00ff00,
  blue = 0x0000ff
}
~~~~

If an explicit discriminator is not specified for a variant, the value
defaults to the value of the previous variant plus one. If the first
variant does not have a discriminator, it defaults to 0. For example,
the value of `north` is 0, `east` is 1, etc.

When an enum is C-like the `as` cast operator can be used to get the
discriminator's value.

<a name="single_variant_enum"></a>

There is a special case for enums with a single variant. These are
used to define new types in such a way that the new name is not just a
synonym for an existing type, but its own distinct type. If you say:

~~~~
enum gizmo_id = int;
~~~~

That is a shorthand for this:

~~~~
enum gizmo_id { gizmo_id(int) }
~~~~

Enum types like this can have their content extracted with the
dereference (`*`) unary operator:

~~~~
# enum gizmo_id = int;
let my_gizmo_id = gizmo_id(10);
let id_int: int = *my_gizmo_id;
~~~~

## Enum patterns

For enum types with multiple variants, destructuring is the only way to
get at their contents. All variant constructors can be used as
patterns, as in this definition of `area`:

~~~~
# type point = {x: float, y: float};
# enum shape { circle(point, float), rectangle(point, point) }
fn area(sh: shape) -> float {
    alt sh {
        circle(_, size) { float::consts::pi * size * size }
        rectangle({x, y}, {x: x2, y: y2}) { (x2 - x) * (y2 - y) }
    }
}
~~~~

Another example, matching nullary enum variants:

~~~~
# type point = {x: float, y: float};
# enum direction { north, east, south, west }
fn point_from_direction(dir: direction) -> point {
    alt dir {
        north { {x:  0f, y:  1f} }
        east  { {x:  1f, y:  0f} }
        south { {x:  0f, y: -1f} }
        west  { {x: -1f, y:  0f} }
    }
}
~~~~

## Tuples

Tuples in Rust behave exactly like records, except that their fields
do not have names (and can thus not be accessed with dot notation).
Tuples can have any arity except for 0 or 1 (though you may consider
nil, `()`, as the empty tuple if you like).

~~~~
let mytup: (int, int, float) = (10, 20, 30.0);
alt mytup {
  (a, b, c) { log(info, a + b + (c as int)); }
}
~~~~

# The Rust memory model

At this junction let's take a detour to explain the concepts involved
in Rust's memory model. Rust has a very particular approach to
memory management that plays a significant role in shaping the "feel"
of the language. Understanding the memory landscape will illuminate
several of Rust's unique features as we encounter them.

Rust has three competing goals that inform its view of memory:

* Memory safety - memory that is managed by and is accessible to
  the Rust language must be guaranteed to be valid. Under normal
  circumstances it is impossible for Rust to trigger a segmentation
  fault or leak memory
* Performance - high-performance low-level code tends to employ
  a number of allocation strategies. low-performance high-level
  code often uses a single, GC-based, heap allocation strategy
* Concurrency - Rust must maintain memory safety guarantees even
  for code running in parallel

## How performance considerations influence the memory model

Many languages that ofter the kinds of memory safety guarentees that
Rust does have a single allocation strategy: objects live on the heap,
live for as long as they are needed, and are periodically garbage
collected. This is very straightforword both conceptually and in
implementation, but has very significant costs. Such languages tend to
aggressively pursue ways to ameliorate allocation costs (think the
Java virtual machine). Rust supports this strategy with _shared
boxes_, memory allocated on the heap that may be referred to (shared)
by multiple variables.

In comparison, languages like C++ offer a very precise control over
where objects are allocated. In particular, it is common to put
them directly on the stack, avoiding expensive heap allocation. In
Rust this is possible as well, and the compiler will use a clever
lifetime analysis to ensure that no variable can refer to stack
objects after they are destroyed.

## How concurrency considerations influence the memory model

Memory safety in a concurrent environment tends to mean avoiding race
conditions between two threads of execution accessing the same
memory. Even high-level languages frequently avoid solving this
problem, requiring programmers to correctly employ locking to unsure
their program is free of races.

Rust starts from the position that memory simply cannot be shared
between tasks. Experience in other languages has proven that isolating
each tasks' heap from each other is a reliable strategy and one that
is easy for programmers to reason about. Having isolated heaps
additionally means that garbage collection must only be done
per-heap. Rust never 'stops the world' to garbage collect memory.

If Rust tasks have completely isolated heaps then that seems to imply
that any data transferred between them must be copied. While this
is a fine and useful way to implement communication between tasks,
it is also very inefficient for large data structures.

Because of this Rust also introduces a global "exchange heap". Objects
allocated here have _ownership semantics_, meaning that there is only
a single variable that refers to them. For this reason they are
refered to as _unique boxes_. All tasks may allocate objects on this
heap, then transfer ownership of those allocations to other tasks,
avoiding expensive copies.

## What to be aware of

Rust has three "realms" in which objects can be allocated: the stack,
the local heap, and the exchange heap. These realms have corresponding
pointer types: the borrowed pointer (`&T`), the shared box (`@T`),
and the unique box (`~T`). These three sigils will appear
repeatedly as we explore the language. Learning the appropriate role
of each is key to using Rust effectively.

# Boxes and pointers

In contrast to a lot of modern languages, aggregate types like records
and enums are not represented as pointers to allocated memory. They
are, like in C and C++, represented directly. This means that if you
`let x = {x: 1f, y: 1f};`, you are creating a record on the stack. If
you then copy it into a data structure, the whole record is copied,
not just a pointer.

For small records like `point`, this is usually more efficient than
allocating memory and going through a pointer. But for big records, or
records with mutable fields, it can be useful to have a single copy on
the heap, and refer to that through a pointer.

Rust supports several types of pointers. The safe pointer types are
`@T` for shared boxes allocated on the local heap, `~T`, for
uniquely-owned boxes allocated on the exchange heap, and `&T`, for
borrowed pointers, which may point to any memory, and whose lifetimes
are governed by the call stack.

Rust also has an unsafe pointer, written `*T`, which is a completely
unchecked pointer type only used in unsafe code (and thus, in typical
Rust code, very rarely).

All pointer types can be dereferenced with the `*` unary operator.

## Shared boxes

Shared boxes are pointers to heap-allocated, garbage collected memory.
Creating a shared box is done by simply applying the unary `@`
operator to an expression. The result of the expression will be boxed,
resulting in a box of the right type. Copying a shared box, as happens
during assignment, only copies a pointer, never the contents of the
box.

~~~~
let x: @int = @10; // New box, refcount of 1
let y = x; // Copy the pointer, increase refcount
// When x and y go out of scope, refcount goes to 0, box is freed
~~~~

Shared boxes never cross task boundaries.

> ***Note:*** shared boxes are currently reclaimed through reference
> counting and cycle collection, but we will switch to a tracing
> garbage collector.

## Unique boxes

In contrast to shared boxes, unique boxes have a single owner and thus
two unique boxes may not refer to the same memory. All unique boxes
across all tasks are allocated on a single _exchange heap_, where
their uniquely owned nature allows them to be passed between tasks.

Because unique boxes are uniquely owned, copying them involves allocating
a new unique box and duplicating the contents. Copying unique boxes
is expensive so the compiler will complain if you do.

~~~~
let x = ~10;
let y = x; // error: copying a non-implicitly copyable type
~~~~

If you really want to copy a unique box you must say so explicitly.

~~~~
let x = ~10;
let y = copy x;
~~~~

This is where the 'move' (`<-`) operator comes in. It is similar to
`=`, but it de-initializes its source. Thus, the unique box can move
from `x` to `y`, without violating the constraint that it only has a
single owner (if you used assignment instead of the move operator, the
box would, in principle, be copied).

~~~~
let x = ~10;
let y <- x;
~~~~

> ***Note:*** this discussion of copying vs moving does not account
> for the "last use" rules that automatically promote copy operations
> to moves. This is an evolving area of the language that will
> continue to change.

Unique boxes, when they do not contain any shared boxes, can be sent
to other tasks. The sending task will give up ownership of the box,
and won't be able to access it afterwards. The receiving task will
become the sole owner of the box.

## Borrowed pointers

Rust borrowed pointers are a general purpose reference/pointer type,
similar to the C++ reference type, but guaranteed to point to valid
memory. In contrast to unique pointers, where the holder of a unique
pointer is the owner of the pointed-to memory, borrowed pointers never
imply ownership. Pointers may be borrowed from any type, in which case
the pointer is guaranteed not to outlive the value it points to.

~~~~
# fn work_with_foo_by_pointer(f: &str) { }
let foo = "foo";
work_with_foo_by_pointer(&foo);
~~~~

The following shows an example of what is _not_ possible with borrowed
pointers. If you were able to write this then the pointer to `foo`
would outlive `foo` itself.

~~~~ {.ignore}
let foo_ptr;
{
    let foo = "foo";
    foo_ptr = &foo;
}
~~~~

> ***Note:*** borrowed pointers are a new addition to the language.
> They are not used extensively yet but are expected to become the
> pointer type used in many common situations, in particular for
> by-reference argument passing. Rust's current solution for passing
> arguments by reference is [argument modes](#argument-passing).

## Mutability

All pointer types have a mutable variant, written `@mut T` or `~mut
T`. Given such a pointer, you can write to its contents by combining
the dereference operator with a mutating action.

~~~~
fn increase_contents(pt: @mut int) {
    *pt += 1;
}
~~~~

# Vectors

Vectors are a contiguous section of memory containing zero or more
values of the same type. Like other types in Rust, vectors can be
stored on the stack, the local heap, or the exchange heap.

~~~
enum crayon {
    almond, antique_brass, apricot,
    aquamarine, asparagus, atomic_tangerine,
    banana_mania, beaver, bittersweet
}

// A stack vector of crayons
let stack_crayons: &[crayon] = &[almond, antique_brass, apricot];
// A local heap (shared) vector of crayons
let local_crayons: @[crayon] = @[aquamarine, asparagus, atomic_tangerine];
// An exchange heap (unique) vector of crayons
let exchange_crayons: ~[crayon] = ~[banana_mania, beaver, bittersweet];
~~~

> ***Note:*** Until recently Rust only had unique vectors, using the
> unadorned `[]` syntax for literals. This syntax is still supported
> but is deprecated. In the future it will probably represent some
> "reasonable default" vector type.
>
> Unique vectors are the currently-recomended vector type for general
> use as they are the most tested and well-supported by existing
> libraries. There will be a gradual shift toward using more
> stack and local vectors in the coming releases.

Vector literals are enclosed in square brackets and dereferencing is
also done with square brackets (zero-based):

~~~~
# enum crayon { almond, antique_brass, apricot,
#               aquamarine, asparagus, atomic_tangerine,
#               banana_mania, beaver, bittersweet };
# fn draw_scene(c: crayon) { }

let crayons = ~[banana_mania, beaver, bittersweet];
if crayons[0] == bittersweet { draw_scene(crayons[0]); }
~~~~

By default, vectors are immutable—you can not replace their elements.
The type written as `~[mut T]` is a vector with mutable
elements. Mutable vector literals are written `~[mut]` (empty) or `~[mut
1, 2, 3]` (with elements).

~~~~
# enum crayon { almond, antique_brass, apricot,
#               aquamarine, asparagus, atomic_tangerine,
#               banana_mania, beaver, bittersweet };

let crayons = ~[mut banana_mania, beaver, bittersweet];
crayons[0] = atomic_tangerine;
~~~~

The `+` operator means concatenation when applied to vector types.

~~~~
# enum crayon { almond, antique_brass, apricot,
#               aquamarine, asparagus, atomic_tangerine,
#               banana_mania, beaver, bittersweet };

let my_crayons = ~[almond, antique_brass, apricot];
let your_crayons = ~[banana_mania, beaver, bittersweet];

let our_crayons = my_crayons + your_crayons;
~~~~

The `+=` operator also works as expected, provided the assignee
lives in a mutable slot.

~~~~
# enum crayon { almond, antique_brass, apricot,
#               aquamarine, asparagus, atomic_tangerine,
#               banana_mania, beaver, bittersweet };

let mut my_crayons = ~[almond, antique_brass, apricot];
let your_crayons = ~[banana_mania, beaver, bittersweet];

my_crayons += your_crayons;
~~~~

## Strings

The `str` type in Rust is represented exactly the same way as a unique
vector of immutable bytes (`~[u8]`). This sequence of bytes is
interpreted as an UTF-8 encoded sequence of characters. This has the
advantage that UTF-8 encoded I/O (which should really be the default
for modern systems) is very fast, and that strings have, for most
intents and purposes, a nicely compact representation. It has the
disadvantage that you only get constant-time access by byte, not by
character.

~~~~
let huh = "what?";
let que: u8 = huh[4]; // indexing a string returns a `u8`
assert que == '?' as u8;
~~~~

A lot of algorithms don't need constant-time indexed access (they
iterate over all characters, which `str::chars` helps with), and
for those that do, many don't need actual characters, and can operate
on bytes. For algorithms that do really need to index by character,
there are core library functions available.

> ***Note:*** like vectors, strings will soon be allocatable in
> the local heap and on the stack, in addition to the exchange heap.

## Vector and string methods

Both vectors and strings support a number of useful
[methods](#implementation).  While we haven't covered methods yet,
most vector functionality is provided by methods, so let's have a
brief look at a few common ones.

~~~
# import io::println;
# enum crayon {
#     almond, antique_brass, apricot,
#     aquamarine, asparagus, atomic_tangerine,
#     banana_mania, beaver, bittersweet
# }
# fn unwrap_crayon(c: crayon) -> int { 0 }
# fn eat_crayon_wax(i: int) { }
# fn store_crayon_in_nasal_cavity(i: uint, c: crayon) { }
# fn crayon_to_str(c: crayon) -> str { "" }

let crayons = ~[almond, antique_brass, apricot];

// Check the length of the vector
assert crayons.len() == 3;
assert !crayons.is_empty();

// Iterate over a vector
for crayons.each |crayon| {
    let delicious_crayon_wax = unwrap_crayon(crayon);
    eat_crayon_wax(delicious_crayon_wax);
}

// Map vector elements
let crayon_names = crayons.map(crayon_to_str);
let favorite_crayon_name = crayon_names[0];

// Remove whitespace from before and after the string
let new_favorite_crayon_name = favorite_crayon_name.trim();

if favorite_crayon_name.len() > 5 {
   // Create a substring
   println(favorite_crayon_name.substr(0, 5));
}
~~~

# Closures

Named functions, like those we've seen so far, may not refer to local
variables decalared outside the function - they do not "close over
their environment". For example you couldn't write the following:

~~~~ {.ignore}
let foo = 10;

fn bar() -> int {
   ret foo; // `bar` cannot refer to `foo`
}
~~~~

Rust also supports _closures_, functions that can access variables in
the enclosing scope.

~~~~
# import println = io::println;
fn call_closure_with_ten(b: fn(int)) { b(10); }

let captured_var = 20;
let closure = |arg| println(#fmt("captured_var=%d, arg=%d", captured_var, arg));

call_closure_with_ten(closure);
~~~~

Closures begin with the argument list between bars and are followed by
a single expression. The types of the arguments are generally omitted,
as is the return type, because the compiler can almost always infer
them. In the rare case where the compiler needs assistance though, the
arguments and return types may be annotated.

~~~~
# type mygoodness = fn(str) -> str; type what_the = int;
let bloop = |well, oh: mygoodness| -> what_the { fail oh(well) };
~~~~

There are several forms of closure, each with its own role. The most
common, called a _stack closure_, has type `fn&` and can directly
access local variables in the enclosing scope.

~~~~
let mut max = 0;
[1, 2, 3].map(|x| if x > max { max = x });
~~~~

Stack closures are very efficient because their environment is
allocated on the call stack and refers by pointer to captured
locals. To ensure that stack closures never outlive the local
variables to which they refer, they can only be used in argument
position and cannot be stored in structures nor returned from
functions. Despite the limitations stack closures are used
pervasively in Rust code.

## Shared closures

When you need to store a closure in a data structure, a stack closure
will not do, since the compiler will refuse to let you store it. For
this purpose, Rust provides a type of closure that has an arbitrary
lifetime, written `fn@` (boxed closure, analogous to the `@` pointer
type described earlier).

A boxed closure does not directly access its environment, but merely
copies out the values that it closes over into a private data
structure. This means that it can not assign to these variables, and
will not 'see' updates to them.

This code creates a closure that adds a given string to its argument,
returns it from a function, and then calls it:

~~~~
use std;

fn mk_appender(suffix: str) -> fn@(str) -> str {
    ret fn@(s: str) -> str { s + suffix };
}

fn main() {
    let shout = mk_appender("!");
    io::println(shout("hey ho, let's go"));
}
~~~~

This example uses the long closure syntax, `fn@(s: str) ...`,
making the fact that we are declaring a box closure explicit. In
practice boxed closures are usually defined with the short closure
syntax introduced earlier, in which case the compiler will infer
the type of closure. Thus our boxed closure example could also
be written:

~~~~
fn mk_appender(suffix: str) -> fn@(str) -> str {
    ret |s| s + suffix;
}
~~~~

## Unique closures

Unique closures, written `fn~` in analogy to the `~` pointer type,
hold on to things that can safely be sent between
processes. They copy the values they close over, much like boxed
closures, but they also 'own' them—meaning no other code can access
them. Unique closures are used in concurrent code, particularly
for spawning [tasks](#tasks).

## Closure compatibility

A nice property of Rust closures is that you can pass any kind of
closure (as long as the arguments and return types match) to functions
that expect a `fn()`. Thus, when writing a higher-order function that
wants to do nothing with its function argument beyond calling it, you
should almost always specify the type of that argument as `fn()`, so
that callers have the flexibility to pass whatever they want.

~~~~
fn call_twice(f: fn()) { f(); f(); }
call_twice(|| { "I am an inferred stack closure"; } );
call_twice(fn&() { "I am also a stack closure"; } );
call_twice(fn@() { "I am a boxed closure"; });
call_twice(fn~() { "I am a unique closure"; });
fn bare_function() { "I am a plain function"; }
call_twice(bare_function);
~~~~

## Do syntax

Closures in Rust are frequently used in combination with higher-order
functions to simulate control structures like `if` and
`loop`. Consider this function that iterates over a vector of
integers, applying an operator to each:

~~~~
fn each(v: ~[int], op: fn(int)) {
   let mut n = 0;
   while n < v.len() {
       op(v[n]);
       n += 1;
   }
}
~~~~

As a caller, if we use a closure to provide the final operator
argument, we can write it in a way that has a pleasant, block-like
structure.

~~~~
# fn each(v: ~[int], op: fn(int)) {}
# fn do_some_work(i: int) { }
each(~[1, 2, 3], |n| {
    #debug("%i", n);
    do_some_work(n);
});
~~~~

This is such a useful pattern that Rust has a special form of function
call that can be written more like a built-in control structure:

~~~~
# fn each(v: ~[int], op: fn(int)) {}
# fn do_some_work(i: int) { }
do each(~[1, 2, 3]) |n| {
    #debug("%i", n);
    do_some_work(n);
}
~~~~

The call is prefixed with the keyword `do` and, instead of writing the
final closure inside the argument list it is moved outside of the
parenthesis where it looks visually more like a typical block of
code. The `do` expression is purely syntactic sugar for a call that
takes a final closure argument.

`do` is often used for task spawning.

~~~~
import task::spawn;

do spawn() || {
    #debug("I'm a task, whatever");
}
~~~~

That's nice, but look at all those bars and parentheses - that's two empty
argument lists back to back. Wouldn't it be great if they weren't
there?

~~~~
# import task::spawn;
do spawn {
   #debug("Kablam!");
}
~~~~

Empty argument lists can be omitted from `do` expressions.

## For loops

Most iteration in Rust is done with `for` loops. Like `do`,
`for` is a nice syntax for doing control flow with closures.
Additionally, within a `for` loop, `break`, `again`, and `ret`
work just as they do with `while` and `loop`.

Consider again our `each` function, this time improved to
break early when the iteratee returns `false`:

~~~~
fn each(v: ~[int], op: fn(int) -> bool) {
   let mut n = 0;
   while n < v.len() {
       if !op(v[n]) {
           break;
       }
       n += 1;
   }
}
~~~~

And using this function to iterate over a vector:

~~~~
# import each = vec::each;
# import println = io::println;
each(~[2, 4, 8, 5, 16], |n| {
    if n % 2 != 0 {
        println("found odd number!");
        false
    } else { true }
});
~~~~

With `for`, functions like `each` can be treated more
like builtin looping structures. When calling `each`
in a `for` loop, instead of returning `false` to break
out of the loop, you just write `break`. To skip ahead
to the next iteration, write `again`.

~~~~
# import each = vec::each;
# import println = io::println;
for each(~[2, 4, 8, 5, 16]) |n| {
    if n % 2 != 0 {
        println("found odd number!");
        break;
    }
}
~~~~

As an added bonus, you can use the `ret` keyword, which is not
normally allowed in closures, in a block that appears as the body of a
`for` loop — this will cause a return to happen from the outer
function, not just the loop body.

~~~~
# import each = vec::each;
fn contains(v: ~[int], elt: int) -> bool {
    for each(v) |x| {
        if (x == elt) { ret true; }
    }
    false
}
~~~~

`for` syntax only works with stack closures.

# Classes

Rust lets users define new types with fields and methods, called 'classes', in
the style of object-oriented languages.

> ***Warning:*** Rust's classes are in the process of changing rapidly. Some more
> information about some of the potential changes is [here][classchanges].

[classchanges]: http://pcwalton.github.com/blog/2012/06/03/maximally-minimal-classes-for-rust/

An example of a class:

~~~~
class example {
  let mut x: int;
  let y: int;

  priv {
    let mut private_member: int;
    fn private_method() {}
  }

  new(x: int) {
    // Constructor
    self.x = x;
    self.y = 7;
    self.private_member = 8;
  }

  fn a() {
    io::println("a");
  }

  drop {
    // Destructor
    self.x = 0;
  }
}

fn main() {
  let x: example = example(1);
  let y: @example = @example(2);
  x.a();
  x.x = 5;
}
~~~~

Fields and methods are declared just like functions and local variables, using
'fn' and 'let'. As usual, 'let mut' can be used to create mutable fields. At
minimum, Rust classes must have at least one field.

Rust classes must also have a constructor, and can optionally have a destructor
as well. The constructor and destructor are declared as shown in the example:
like methods named 'new' and 'drop', but without 'fn', and without arguments
for drop.

In the constructor, the compiler will enforce that all fields are initialized
before doing anything which might allow them to be accessed. This includes
returning from the constructor, calling any method on 'self', calling any
function with 'self' as an argument, or taking a reference to 'self'. Mutation
of immutable fields is possible only in the constructor, and only before doing
any of these things; afterwards it is an error.

Private fields and methods are declared as shown above, using a `priv { ... }`
block within the class. They are accessible only from within the same instance
of the same class. (For example, even from within class A, you cannot call
private methods, or access private fields, on other instances of class A; only
on `self`.) This accessibility restriction may change in the future.

As mentioned below, in the section on copying types, classes with destructors
are considered 'resource' types and are not copyable.

Declaring a class also declares its constructor as a function of the same name.
You can construct an instance of the class, as in the example, by calling that
function. The function and the type, though they have the same name, are
otherwise independent. As with other Rust types, you can use `@` or `~` to
construct a heap-allocated instance of a class, either shared or unique; just
call e.g. `@example(...)` as shown above.

# Argument passing

Rust datatypes are not trivial to copy (the way, for example,
JavaScript values can be copied by simply taking one or two machine
words and plunking them somewhere else). Shared boxes require
reference count updates, and big records, enums, or unique pointers require
an arbitrary amount of data to be copied (plus updating the reference
counts of shared boxes hanging off them).

For this reason, the default calling convention for Rust functions
leaves ownership of the arguments with the caller. The caller
guarantees that the arguments will outlive the call, the callee merely
gets access to them.

## Safe references

*This system has recently changed.  An explanantion is forthcoming.*

## Other uses of safe references

Safe references are not only used for argument passing. When you
destructure on a value in an `alt` expression, or loop over a vector
with `for`, variables bound to the inside of the given data structure
will use safe references, not copies. This means such references are
very cheap, but you'll occasionally have to copy them to ensure
safety.

~~~~
let mut my_rec = {a: 4, b: ~[1, 2, 3]};
alt my_rec {
  {a, b} {
    log(info, b); // This is okay
    my_rec = {a: a + 1, b: b + ~[a]};
    log(info, b); // Here reference b has become invalid
  }
}
~~~~

## Argument passing styles

The fact that arguments are conceptually passed by safe reference does
not mean all arguments are passed by pointer. Composite types like
records and enums *are* passed by pointer, but single-word values, like
integers and pointers, are simply passed by value. Most of the time,
the programmer does not have to worry about this, as the compiler will
simply pick the most efficient passing style. There is one exception,
which will be described in the section on [generics](#generics).

To explicitly set the passing-style for a parameter, you prefix the
argument name with a sigil. There are three special passing styles that
are often useful. The first is by-mutable-pointer, written with a
single `&`:

~~~~
fn vec_push(&v: ~[int], elt: int) {
    v += ~[elt];
}
~~~~

This allows the function to mutate the value of the argument, *in the
caller's context*. Clearly, you are only allowed to pass things that
can actually be mutated to such a function.

Then there is the by-copy style, written `+`. This indicates that the
function wants to take ownership of the argument value. If the caller
does not use the argument after the call, it will be 'given' to the
callee. Otherwise a copy will be made. This mode is mostly used for
functions that construct data structures. The argument will end up
being owned by the data structure, so if that can be done without a
copy, that's a win.

~~~~
type person = {name: str, address: str};
fn make_person(+name: str, +address: str) -> person {
    ret {name: name, address: address};
}
~~~~

Finally there is by-move style, written `-`. This indicates that the
function will take ownership of the argument, like with by-copy style,
but a copy must not be made. The caller is (statically) obliged to not
use the argument after the call; it is de-initialized as part of the
call. This is used to support ownership-passing in the presence of
non-copyable types.

# Generics

## Generic functions

Throughout this tutorial, we've been defining functions like
that act only on single data types. It is 2012, and we no longer
expect to be defining such functions again and again for every type
they apply to.  Thus, Rust allows functions and datatypes to have type
parameters.

~~~~
fn map<T, U>(vector: ~[T], function: fn(T) -> U) -> ~[U] {
    let mut accumulator = ~[];
    for vector.each |element| {
        vec::push(accumulator, function(element));
    }
    ret accumulator;
}
~~~~

When defined with type parameters, this function can be applied to any
type of vector, as long as the type of `function`'s argument and the
type of the vector's content agree with each other.

Inside a generic function, the names of the type parameters
(capitalized by convention) stand for opaque types. You can't look
inside them, but you can pass them around.

## Generic datatypes

Generic `type` and `enum` declarations follow the same pattern:

~~~~
type circular_buf<T> = {start: uint,
                        end: uint,
                        buf: ~[mut T]};

enum option<T> { some(T), none }
~~~~

You can then declare a function to take a `circular_buf<u8>` or return
an `option<str>`, or even an `option<T>` if the function itself is
generic.

The `option` type given above exists in the core library and is the
way Rust programs express the thing that in C would be a nullable
pointer. The nice part is that you have to explicitly unpack an
`option` type, so accidental null pointer dereferences become
impossible.

## Type-inference and generics

Rust's type inferrer works very well with generics, but there are
programs that just can't be typed.

~~~~
let n = option::none;
# option::iter(n, fn&(&&x:int) {})
~~~~

If you never do anything else with `n`, the compiler will not be able
to assign a type to it. (The same goes for `[]`, the empty vector.) If
you really want to have such a statement, you'll have to write it like
this:

~~~~
let n2: option<int> = option::none;
// or
let n = option::none::<int>;
~~~~

Note that, in a value expression, `<` already has a meaning as a
comparison operator, so you'll have to write `::<T>` to explicitly
give a type to a name that denotes a generic value. Fortunately, this
is rarely necessary.

## Polymorphic built-ins

There are two built-in operations that, perhaps surprisingly, act on
values of any type. It was already mentioned earlier that `log` can
take any type of value and output it.

More interesting is that Rust also defines an ordering for values of
all datatypes, and allows you to meaningfully apply comparison
operators (`<`, `>`, `<=`, `>=`, `==`, `!=`) to them. For structural
types, the comparison happens left to right, so `"abc" < "bac"` (but
note that `"bac" < "ác"`, because the ordering acts on UTF-8 sequences
without any sophistication).

## Kinds

Perhaps surprisingly, the 'copy' (duplicate) operation is not defined
for all Rust types. Resource types (classes with destructors) cannot be
copied, and neither can any type whose copying would require copying a
resource (such as records or unique boxes containing a resource).

This complicates handling of generic functions. If you have a type
parameter `T`, can you copy values of that type? In Rust, you can't,
unless you explicitly declare that type parameter to have copyable
'kind'. A kind is a type of type.

~~~~ {.ignore}
// This does not compile
fn head_bad<T>(v: ~[T]) -> T { v[0] }
// This does
fn head<T: copy>(v: ~[T]) -> T { v[0] }
~~~~

When instantiating a generic function, you can only instantiate it
with types that fit its kinds. So you could not apply `head` to a
resource type. Rust has several kinds that can be used as type bounds:

* `copy` - Copyable types. All types are copyable unless they
  are classes with destructors or otherwise contain
  classes with destructors.
* `send` - Sendable types. All types are sendable unless they
  contain shared boxes, closures, or other local-heap-allocated
  types.
* `const` - Constant types. These are types that do not contain
  mutable fields nor shared boxes.

> ***Note:*** Rust type kinds are syntactically very similar to
> [interfaces](#interfaces) when used as type bounds, and can be
> conveniently thought of as built-in interfaces. In the future type
> kinds will actually be interfaces that the compiler has special
> knowledge about.

## Generic functions and argument-passing

The previous section mentioned that arguments are passed by pointer or
by value based on their type. There is one situation in which this is
difficult. If you try this program:

~~~~{.xfail-test}
fn plus1(x: int) -> int { x + 1 }
vec::map(~[1, 2, 3], plus1);
~~~~

You will get an error message about argument passing styles
disagreeing. The reason is that generic types are always passed by
reference, so `map` expects a function that takes its argument by
reference. The `plus1` you defined, however, uses the default,
efficient way to pass integers, which is by value. To get around this
issue, you have to explicitly mark the arguments to a function that
you want to pass to a generic higher-order function as being passed by
pointer, using the `&&` sigil:

~~~~
fn plus1(&&x: int) -> int { x + 1 }
vec::map(~[1, 2, 3], plus1);
~~~~

> ***Note:*** This is inconvenient, and we are hoping to get rid of
> this restriction in the future.

# Modules and crates

The Rust namespace is divided into modules. Each source file starts
with its own module.

## Local modules

The `mod` keyword can be used to open a new, local module. In the
example below, `chicken` lives in the module `farm`, so, unless you
explicitly import it, you must refer to it by its long name,
`farm::chicken`.

~~~~
mod farm {
    fn chicken() -> str { "cluck cluck" }
    fn cow() -> str { "mooo" }
}
fn main() {
    io::println(farm::chicken());
}
~~~~

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

~~~~ {.ignore}
#[link(name = "farm", vers = "2.5", author = "mjh")];
#[crate_type = "lib"];
mod cow;
mod chicken;
mod horse;
~~~~

Compiling this file will cause `rustc` to look for files named
`cow.rs`, `chicken.rs`, `horse.rs` in the same directory as the `.rc`
file, compile them all together, and, depending on the presence of the
`crate_type = "lib"` attribute, output a shared library or an executable.
(If the line `#[crate_type = "lib"];` was omitted, `rustc` would create an
executable.)

The `#[link(...)]` part provides meta information about the module,
which other crates can use to load the right module. More about that
later.

To have a nested directory structure for your source files, you can
nest mods in your `.rc` file:

~~~~ {.ignore}
mod poultry {
    mod chicken;
    mod turkey;
}
~~~~

The compiler will now look for `poultry/chicken.rs` and
`poultry/turkey.rs`, and export their content in `poultry::chicken`
and `poultry::turkey`. You can also provide a `poultry.rs` to add
content to the `poultry` module itself.

## Using other crates

Having compiled a crate that contains the `#[crate_type = "lib"]` attribute,
you can use it in another crate with a `use` directive. We've already seen
`use std` in several of the examples, which loads in the [standard library][std].

[std]: http://doc.rust-lang.org/doc/std/index/General.html

`use` directives can appear in a crate file, or at the top level of a
single-file `.rs` crate. They will cause the compiler to search its
library search path (which you can extend with `-L` switch) for a Rust
crate library with the right name.

It is possible to provide more specific information when using an
external crate.

~~~~ {.ignore}
use myfarm (name = "farm", vers = "2.7");
~~~~

When a comma-separated list of name/value pairs is given after `use`,
these are matched against the attributes provided in the `link`
attribute of the crate file, and a crate is only used when the two
match. A `name` value can be given to override the name used to search
for the crate. So the above would import the `farm` crate under the
local name `myfarm`.

Our example crate declared this set of `link` attributes:

~~~~ {.ignore}
#[link(name = "farm", vers = "2.5", author = "mjh")];
~~~~

The version does not match the one provided in the `use` directive, so
unless the compiler can find another crate with the right version
somewhere, it will complain that no matching crate was found.

## The core library

A set of basic library routines, mostly related to built-in datatypes
and the task system, are always implicitly linked and included in any
Rust program.

This library is documented [here][core].

[core]: http://doc.rust-lang.org/doc/core

## A minimal example

Now for something that you can actually compile yourself. We have
these two files:

~~~~
// mylib.rs
#[link(name = "mylib", vers = "1.0")];
fn world() -> str { "world" }
~~~~

~~~~ {.ignore}
// main.rs
use std;
use mylib;
fn main() { io::println("hello " + mylib::world()); }
~~~~

Now compile and run like this (adjust to your platform if necessary):

~~~~ {.notrust}
> rustc --lib mylib.rs
> rustc main.rs -L .
> ./main
"hello world"
~~~~

## Importing

When using identifiers from other modules, it can get tiresome to
qualify them with the full module path every time (especially when
that path is several modules deep). Rust allows you to import
identifiers at the top of a file, module, or block.

~~~~
use std;
import io::println;
fn main() {
    println("that was easy");
}
~~~~

It is also possible to import just the name of a module (`import
std::list;`, then use `list::find`), to import all identifiers exported
by a given module (`import io::*`), or to import a specific set
of identifiers (`import math::{min, max, pi}`).

You can rename an identifier when importing using the `=` operator:

~~~~
import prnt = io::println;
~~~~

## Exporting

By default, a module exports everything that it defines. This can be
restricted with `export` directives at the top of the module or file.

~~~~
mod enc {
    export encrypt, decrypt;
    const super_secret_number: int = 10;
    fn encrypt(n: int) -> int { n + super_secret_number }
    fn decrypt(n: int) -> int { n - super_secret_number }
}
~~~~

This defines a rock-solid encryption algorithm. Code outside of the
module can refer to the `enc::encrypt` and `enc::decrypt` identifiers
just fine, but it does not have access to `enc::super_secret_number`.

## Namespaces

Rust uses three different namespaces: one for modules, one for types,
and one for values. This means that this code is valid:

~~~~
mod buffalo {
    type buffalo = int;
    fn buffalo(buffalo: buffalo) -> buffalo { buffalo }
}
fn main() {
    let buffalo: buffalo::buffalo = 1;
    buffalo::buffalo(buffalo::buffalo(buffalo));
}
~~~~

You don't want to write things like that, but it *is* very practical
to not have to worry about name clashes between types, values, and
modules. This allows us to have a module `core::str`, for example, even
though `str` is a built-in type name.

## Resolution

The resolution process in Rust simply goes up the chain of contexts,
looking for the name in each context. Nested functions and modules
create new contexts inside their parent function or module. A file
that's part of a bigger crate will have that crate's context as its
parent context.

Identifiers can shadow each other. In this program, `x` is of type
`int`:

~~~~
type t = str;
fn main() {
    type t = int;
    let x: t;
}
~~~~

An `import` directive will only import into the namespaces for which
identifiers are actually found. Consider this example:

~~~~
type bar = uint;
mod foo { fn bar() {} }
mod baz {
    import foo::bar;
    const x: bar = 20u;
}
~~~~

When resolving the type name `bar` in the `const` definition, the
resolver will first look at the module context for `baz`. This has an
import named `bar`, but that's a function, not a type, So it continues
to the top level and finds a type named `bar` defined there.

Normally, multiple definitions of the same identifier in a scope are
disallowed. Local variables defined with `let` are an exception to
this—multiple `let` directives can redefine the same variable in a
single scope. When resolving the name of such a variable, the most
recent definition is used.

~~~~
fn main() {
    let x = 10;
    let x = x + 10;
    assert x == 20;
}
~~~~

This makes it possible to rebind a variable without actually mutating
it, which is mostly useful for destructuring (which can rebind, but
not assign).

# Interfaces

Interfaces are Rust's take on value polymorphism—the thing that
object-oriented languages tend to solve with methods and inheritance.
For example, writing a function that can operate on multiple types of
collections.

> ***Note:*** This feature is very new, and will need a few extensions to be
> applicable to more advanced use cases.

## Declaration

An interface consists of a set of methods. A method is a function that
can be applied to a `self` value and a number of arguments, using the
dot notation: `self.foo(arg1, arg2)`.

For example, we could declare the interface `to_str` for things that
can be converted to a string, with a single method of the same name:

~~~~
iface to_str {
    fn to_str() -> str;
}
~~~~

## Implementation

To actually implement an interface for a given type, the `impl` form
is used. This defines implementations of `to_str` for the `int` and
`str` types.

~~~~
# iface to_str { fn to_str() -> str; }
impl of to_str for int {
    fn to_str() -> str { int::to_str(self, 10u) }
}
impl of to_str for str {
    fn to_str() -> str { self }
}
~~~~

Given these, we may call `1.to_str()` to get `"1"`, or
`"foo".to_str()` to get `"foo"` again. This is basically a form of
static overloading—when the Rust compiler sees the `to_str` method
call, it looks for an implementation that matches the type with a
method that matches the name, and simply calls that.

## Scoping

Implementations are not globally visible. Resolving a method to an
implementation requires that implementation to be in scope. You can
import and export implementations using the name of the interface they
implement (multiple implementations with the same name can be in scope
without problems). Or you can give them an explicit name if you
prefer, using this syntax:

~~~~
# iface to_str { fn to_str() -> str; }
impl nil_to_str of to_str for () {
    fn to_str() -> str { "()" }
}
~~~~

## Bounded type parameters

The useful thing about value polymorphism is that it does not have to
be static. If object-oriented languages only let you call a method on
an object when they knew exactly which sub-type it had, that would not
get you very far. To be able to call methods on types that aren't
known at compile time, it is possible to specify 'bounds' for type
parameters.

~~~~
# iface to_str { fn to_str() -> str; }
fn comma_sep<T: to_str>(elts: ~[T]) -> str {
    let mut result = "", first = true;
    for elts.each |elt| {
        if first { first = false; }
        else { result += ", "; }
        result += elt.to_str();
    }
    ret result;
}
~~~~

The syntax for this is similar to the syntax for specifying that a
parameter type has to be copyable (which is, in principle, another
kind of bound). By declaring `T` as conforming to the `to_str`
interface, it becomes possible to call methods from that interface on
values of that type inside the function. It will also cause a
compile-time error when anyone tries to call `comma_sep` on an array
whose element type does not have a `to_str` implementation in scope.

## Polymorphic interfaces

Interfaces may contain type parameters. This defines an interface for
generalized sequence types:

~~~~
iface seq<T> {
    fn len() -> uint;
    fn iter(fn(T));
}
impl <T> of seq<T> for ~[T] {
    fn len() -> uint { vec::len(self) }
    fn iter(b: fn(T)) {
        for self.each |elt| { b(elt); }
    }
}
~~~~

Note that the implementation has to explicitly declare the its
parameter `T` before using it to specify its interface type. This is
needed because it could also, for example, specify an implementation
of `seq<int>`—the `of` clause *refers* to a type, rather than defining
one.

## Casting to an interface type

The above allows us to define functions that polymorphically act on
values of *an* unknown type that conforms to a given interface.
However, consider this function:

~~~~
# type circle = int; type rectangle = int;
# iface drawable { fn draw(); }
# impl of drawable for int { fn draw() {} }
# fn new_circle() -> int { 1 }
fn draw_all<T: drawable>(shapes: ~[T]) {
    for shapes.each |shape| { shape.draw(); }
}
# let c: circle = new_circle();
# draw_all(~[c]);
~~~~

You can call that on an array of circles, or an array of squares
(assuming those have suitable `drawable` interfaces defined), but not
on an array containing both circles and squares.

When this is needed, an interface name can be used as a type, causing
the function to be written simply like this:

~~~~
# iface drawable { fn draw(); }
fn draw_all(shapes: ~[drawable]) {
    for shapes.each |shape| { shape.draw(); }
}
~~~~

There is no type parameter anymore (since there isn't a single type
that we're calling the function on). Instead, the `drawable` type is
used to refer to a type that is a reference-counted box containing a
value for which a `drawable` implementation exists, combined with
information on where to find the methods for this implementation. This
is very similar to the 'vtables' used in most object-oriented
languages.

To construct such a value, you use the `as` operator to cast a value
to an interface type:

~~~~
# type circle = int; type rectangle = int;
# iface drawable { fn draw(); }
# impl of drawable for int { fn draw() {} }
# fn new_circle() -> int { 1 }
# fn new_rectangle() -> int { 2 }
# fn draw_all(shapes: ~[drawable]) {}
let c: circle = new_circle();
let r: rectangle = new_rectangle();
draw_all(~[c as drawable, r as drawable]);
~~~~

This will store the value into a box, along with information about the
implementation (which is looked up in the scope of the cast). The
`drawable` type simply refers to such boxes, and calling methods on it
always works, no matter what implementations are in scope.

Note that the allocation of a box is somewhat more expensive than
simply using a type parameter and passing in the value as-is, and much
more expensive than statically resolved method calls.

## Interface-less implementations

If you only intend to use an implementation for static overloading,
and there is no interface available that it conforms to, you are free
to leave off the `of` clause.

~~~~
# type currency = ();
# fn mk_currency(x: int, s: str) {}
impl int_util for int {
    fn times(b: fn(int)) {
        let mut i = 0;
        while i < self { b(i); i += 1; }
    }
    fn dollars() -> currency {
        mk_currency(self, "USD")
    }
}
~~~~

This allows cutesy things like `send_payment(10.dollars())`. And the
nice thing is that it's fully scoped, so the uneasy feeling that
anybody with experience in object-oriented languages (with the
possible exception of Rubyists) gets at the sight of such things is
not justified. It's harmless!

# Interacting with foreign code

One of Rust's aims, as a system programming language, is to
interoperate well with C code.

We'll start with an example. It's a bit bigger than usual, and
contains a number of new concepts. We'll go over it one piece at a
time.

This is a program that uses OpenSSL's `SHA1` function to compute the
hash of its first command-line argument, which it then converts to a
hexadecimal string and prints to standard output. If you have the
OpenSSL libraries installed, it should 'just work'.

~~~~ {.xfail-test}
use std;

extern mod crypto {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}

fn as_hex(data: ~[u8]) -> str {
    let mut acc = "";
    for data.each |byte| { acc += #fmt("%02x", byte as uint); }
    ret acc;
}

fn sha1(data: str) -> str unsafe {
    let bytes = str::bytes(data);
    let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                            vec::len(bytes), ptr::null());
    ret as_hex(vec::unsafe::from_buf(hash, 20u));
}

fn main(args: ~[str]) {
    io::println(sha1(args[1]));
}
~~~~

## Foreign modules

Before we can call `SHA1`, we have to declare it. That is what this
part of the program is responsible for:

~~~~ {.xfail-test}
extern mod crypto {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}
~~~~

An `extern` module declaration containing function signatures introduces
the functions listed as _foreign functions_, that are implemented in some
other language (usually C) and accessed through Rust's foreign function
interface (FFI). An extern module like this is called a foreign module, and
implicitly tells the compiler to link with a library with the same name as
the module, and that it will find the foreign functions in that library.

In this case, it'll change the name `crypto` to a shared library name
in a platform-specific way (`libcrypto.so` on Linux, for example), and
link that in. If you want the module to have a different name from the
actual library, you can use the `"link_name"` attribute, like:

~~~~ {.xfail-test}
#[link_name = "crypto"]
extern mod something {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}
~~~~

## Foreign calling conventions

Most foreign code will be C code, which usually uses the `cdecl` calling
convention, so that is what Rust uses by default when calling foreign
functions. Some foreign functions, most notably the Windows API, use other
calling conventions, so Rust provides a way to hint to the compiler which
is expected by using the `"abi"` attribute:

~~~~
#[cfg(target_os = "win32")]
#[abi = "stdcall"]
extern mod kernel32 {
    fn SetEnvironmentVariableA(n: *u8, v: *u8) -> int;
}
~~~~

The `"abi"` attribute applies to a foreign module (it can not be applied
to a single function within a module), and must be either `"cdecl"`
or `"stdcall"`. Other conventions may be defined in the future.

## Unsafe pointers

The foreign `SHA1` function is declared to take three arguments, and
return a pointer.

~~~~ {.xfail-test}
# extern mod crypto {
fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
# }
~~~~

When declaring the argument types to a foreign function, the Rust
compiler has no way to check whether your declaration is correct, so
you have to be careful. If you get the number or types of the
arguments wrong, you're likely to get a segmentation fault. Or,
probably even worse, your code will work on one platform, but break on
another.

In this case, `SHA1` is defined as taking two `unsigned char*`
arguments and one `unsigned long`. The rust equivalents are `*u8`
unsafe pointers and an `uint` (which, like `unsigned long`, is a
machine-word-sized type).

Unsafe pointers can be created through various functions in the
standard lib, usually with `unsafe` somewhere in their name. You can
dereference an unsafe pointer with `*` operator, but use
caution—unlike Rust's other pointer types, unsafe pointers are
completely unmanaged, so they might point at invalid memory, or be
null pointers.

## Unsafe blocks

The `sha1` function is the most obscure part of the program.

~~~~
# mod crypto { fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8 { out } }
# fn as_hex(data: ~[u8]) -> str { "hi" }
fn sha1(data: str) -> str {
    unsafe {
        let bytes = str::bytes(data);
        let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                                vec::len(bytes), ptr::null());
        ret as_hex(vec::unsafe::from_buf(hash, 20u));
    }
}
~~~~

Firstly, what does the `unsafe` keyword at the top of the function
mean? `unsafe` is a block modifier—it declares the block following it
to be known to be unsafe.

Some operations, like dereferencing unsafe pointers or calling
functions that have been marked unsafe, are only allowed inside unsafe
blocks. With the `unsafe` keyword, you're telling the compiler 'I know
what I'm doing'. The main motivation for such an annotation is that
when you have a memory error (and you will, if you're using unsafe
constructs), you have some idea where to look—it will most likely be
caused by some unsafe code.

Unsafe blocks isolate unsafety. Unsafe functions, on the other hand,
advertise it to the world. An unsafe function is written like this:

~~~~
unsafe fn kaboom() { "I'm harmless!"; }
~~~~

This function can only be called from an unsafe block or another
unsafe function.

## Pointer fiddling

The standard library defines a number of helper functions for dealing
with unsafe data, casting between types, and generally subverting
Rust's safety mechanisms.

Let's look at our `sha1` function again.

~~~~
# mod crypto { fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8 { out } }
# fn as_hex(data: ~[u8]) -> str { "hi" }
# fn x(data: str) -> str {
# unsafe {
let bytes = str::bytes(data);
let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                        vec::len(bytes), ptr::null());
ret as_hex(vec::unsafe::from_buf(hash, 20u));
# }
# }
~~~~

The `str::bytes` function is perfectly safe, it converts a string to
an `[u8]`. This byte array is then fed to `vec::unsafe::to_ptr`, which
returns an unsafe pointer to its contents.

This pointer will become invalid as soon as the vector it points into
is cleaned up, so you should be very careful how you use it. In this
case, the local variable `bytes` outlives the pointer, so we're good.

Passing a null pointer as third argument to `SHA1` causes it to use a
static buffer, and thus save us the effort of allocating memory
ourselves. `ptr::null` is a generic function that will return an
unsafe null pointer of the correct type (Rust generics are awesome
like that—they can take the right form depending on the type that they
are expected to return).

Finally, `vec::unsafe::from_buf` builds up a new `[u8]` from the
unsafe pointer that was returned by `SHA1`. SHA1 digests are always
twenty bytes long, so we can pass `20u` for the length of the new
vector.

## Passing structures

C functions often take pointers to structs as arguments. Since Rust
records are binary-compatible with C structs, Rust programs can call
such functions directly.

This program uses the Posix function `gettimeofday` to get a
microsecond-resolution timer.

~~~~
use std;
type timeval = {mut tv_sec: uint,
                mut tv_usec: uint};
#[nolink]
extern mod libc {
    fn gettimeofday(tv: *timeval, tz: *()) -> i32;
}
fn unix_time_in_microseconds() -> u64 unsafe {
    let x = {mut tv_sec: 0u, mut tv_usec: 0u};
    libc::gettimeofday(ptr::addr_of(x), ptr::null());
    ret (x.tv_sec as u64) * 1000_000_u64 + (x.tv_usec as u64);
}

# fn main() { assert #fmt("%?", unix_time_in_microseconds()) != ""; }
~~~~

The `#[nolink]` attribute indicates that there's no foreign library to
link in. The standard C library is already linked with Rust programs.

A `timeval`, in C, is a struct with two 32-bit integers. Thus, we
define a record type with the same contents, and declare
`gettimeofday` to take a pointer to such a record.

The second argument to `gettimeofday` (the time zone) is not used by
this program, so it simply declares it to be a pointer to the nil
type. Since null pointer look the same, no matter which type they are
supposed to point at, this is safe.

# Tasks

Rust supports a system of lightweight tasks, similar to what is found
in Erlang or other actor systems. Rust tasks communicate via messages
and do not share data. However, it is possible to send data without
copying it by making use of [the exchange heap](#unique-boxes), which
allow the sending task to release ownership of a value, so that the
receiving task can keep on using it.

> ***Note:*** As Rust evolves, we expect the task API to grow and
> change somewhat.  The tutorial documents the API as it exists today.

## Spawning a task

Spawning a task is done using the various spawn functions in the
module `task`.  Let's begin with the simplest one, `task::spawn()`:

~~~~
import task::spawn;
import io::println;

let some_value = 22;

do spawn {
    println("This executes in the child task.");
    println(#fmt("%d", some_value));
}
~~~~

The argument to `task::spawn()` is a [unique
closure](#unique-closures) of type `fn~()`, meaning that it takes no
arguments and generates no return value. The effect of `task::spawn()`
is to fire up a child task that will execute the closure in parallel
with the creator.

## Ports and channels

Now that we have spawned a child task, it would be nice if we could
communicate with it.  This is done by creating a *port* with an
associated *channel*.  A port is simply a location to receive messages
of a particular type.  A channel is used to send messages to a port.
For example, imagine we wish to perform two expensive computations
in parallel.  We might write something like:

~~~~
import task::spawn;
import comm::{port, chan, methods};

let port = port();
let chan = port.chan();

do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}

some_other_expensive_computation();
let result = port.recv();

# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
~~~~

Let's walk through this code line-by-line.  The first line creates a
port for receiving integers:

~~~~ {.ignore}
# import comm::port;
let port = port();
~~~~

This port is where we will receive the message from the child task
once it is complete.  The second line creates a channel for sending
integers to the port `port`:

~~~~
# import comm::{port, chan, methods};
# let port = port::<int>();
let chan = port.chan();
~~~~

The channel will be used by the child to send a message to the port.
The next statement actually spawns the child:

~~~~
# import task::{spawn};
# import comm::{port, chan, methods};
# fn some_expensive_computation() -> int { 42 }
# let port = port();
# let chan = port.chan();
do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}
~~~~

This child will perform the expensive computation send the result
over the channel.  Finally, the parent continues by performing
some other expensive computation and then waiting for the child's result
to arrive on the port:

~~~~
# import comm::{port, chan, methods};
# fn some_other_expensive_computation() {}
# let port = port::<int>();
# let chan = chan::<int>(port);
# chan.send(0);
some_other_expensive_computation();
let result = port.recv();
~~~~

## Creating a task with a bi-directional communication path

A very common thing to do is to spawn a child task where the parent
and child both need to exchange messages with each
other. The function `task::spawn_listener()` supports this pattern. We'll look
briefly at how it is used.

To see how `spawn_listener()` works, we will create a child task
which receives `uint` messages, converts them to a string, and sends
the string in response.  The child terminates when `0` is received.
Here is the function which implements the child task:

~~~~
# import comm::{port, chan, methods};
fn stringifier(from_parent: port<uint>,
               to_parent: chan<str>) {
    let mut value: uint;
    loop {
        value = from_parent.recv();
        to_parent.send(uint::to_str(value, 10u));
        if value == 0u { break; }
    }
}
~~~~

You can see that the function takes two parameters.  The first is a
port used to receive messages from the parent, and the second is a
channel used to send messages to the parent.  The body itself simply
loops, reading from the `from_parent` port and then sending its
response to the `to_parent` channel.  The actual response itself is
simply the strified version of the received value,
`uint::to_str(value)`.
 
Here is the code for the parent task:

~~~~
# import task::{spawn_listener};
# import comm::{chan, port, methods};
# fn stringifier(from_parent: comm::port<uint>,
#                to_parent: comm::chan<str>) {
#     comm::send(to_parent, "22");
#     comm::send(to_parent, "23");
#     comm::send(to_parent, "0");
# }
# fn main() {

let from_child = port();
let to_parent = from_child.chan();
let to_child = do spawn_listener |from_parent| {
    stringifier(from_parent, to_parent);
};

to_child.send(22u);
assert from_child.recv() == "22";

to_child.send(23u);
assert from_child.recv() == "23";

to_child.send(0u);
assert from_child.recv() == "0";

# }
~~~~

The parent first sets up a port to receive data from and a channel
that the child can use to send data to that port. The call to
`spawn_listener()` will spawn the child task, providing it with a port
on which to receive data from its parent, and returning to the parent
the associated channel. Finally, the closure passed to
`spawn_listener()` that forms the body of the child task captures the
`to_parent` channel in its environment, so both parent and child
can send and receive data to and from the other.

# Testing

The Rust language has a facility for testing built into the language.
Tests can be interspersed with other code, and annotated with the
`#[test]` attribute.

~~~~{.xfail-test}
# // FIXME: xfailed because test_twice is a #[test] function it's not
# // getting compiled
use std;

fn twice(x: int) -> int { x + x }

#[test]
fn test_twice() {
    let mut i = -100;
    while i < 100 {
        assert twice(i) == 2 * i;
        i += 1;
    }
}
~~~~

When you compile the program normally, the `test_twice` function will
not be included. To compile and run such tests, compile with the
`--test` flag, and then run the result:

~~~~ {.notrust}
> rustc --test twice.rs
> ./twice
running 1 tests
test test_twice ... ok
result: ok. 1 passed; 0 failed; 0 ignored
~~~~

Or, if we change the file to fail, for example by replacing `x + x`
with `x + 1`:

~~~~ {.notrust}
running 1 tests
test test_twice ... FAILED
failures:
    test_twice
result: FAILED. 0 passed; 1 failed; 0 ignored
~~~~

You can pass a command-line argument to a program compiled with
`--test` to run only the tests whose name matches the given string. If
we had, for example, test functions `test_twice`, `test_once_1`, and
`test_once_2`, running our program with `./twice test_once` would run
the latter two, and running it with `./twice test_once_2` would run
only the last.

To indicate that a test is supposed to fail instead of pass, you can
give it a `#[should_fail]` attribute.

~~~~
use std;

fn divide(a: float, b: float) -> float {
    if b == 0f { fail; }
    a / b
}

#[test]
#[should_fail]
fn divide_by_zero() { divide(1f, 0f); }

# fn main() { }
~~~~

To disable a test completely, add an `#[ignore]` attribute. Running a
test runner (the program compiled with `--test`) with an `--ignored`
command-line flag will cause it to also run the tests labelled as
ignored.

A program compiled as a test runner will have the configuration flag
`test` defined, so that you can add code that won't be included in a
normal compile with the `#[cfg(test)]` attribute (see [conditional
compilation](#attributes)).
