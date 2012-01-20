% Rust Language Tutorial

# Introduction

## Scope

This is a tutorial for the Rust programming language. It assumes the
reader is familiar with the basic concepts of programming, and has
programmed in one or more other languages before. The tutorial covers
the whole language, though not with the depth and precision of the
[language reference](rust.html).

## Disclaimer

Rust is a language under development. The general flavor of the
language has settled, but details will continue to change as it is
further refined. Nothing in this tutorial is final, and though we try
to keep it updated, it is possible that the text occasionally does not
reflect the actual state of the language.

## First Impressions

Though syntax is something you get used to, an initial encounter with
a language can be made easier if the notation looks familiar. Rust is
a curly-brace language in the tradition of C, C++, and JavaScript.

~~~~
fn fac(n: int) -> int {
    let result = 1, i = 1;
    while i <= n {
        result *= i;
        i += 1;
    }
    ret result;
}
~~~~

Several differences from C stand out. Types do not come before, but
after variable names (preceded by a colon). In local variables
(introduced with `let`), they are optional, and will be inferred when
left off. Constructs like `while` and `if` do not require parentheses
around the condition (though they allow them). Also, there's a
tendency towards aggressive abbreviation in the keywords—`fn` for
function, `ret` for return.

You should, however, not conclude that Rust is simply an evolution of
C. As will become clear in the rest of this tutorial, it goes in
quite a different direction.

## Conventions

Throughout the tutorial, words that indicate language keywords or
identifiers defined in the example code are displayed in `code font`.

Code snippets are indented, and also shown in a monospaced font. Not
all snippets constitute whole programs. For brevity, we'll often show
fragments of programs that don't compile on their own. To try them
out, you might have to wrap them in `fn main() { ... }`, and make sure
they don't contain references to things that aren't actually defined.

# Getting started

## Installation

On win32, we make an executable [installer][] available. On other
platforms you need to build from a [tarball][].

If you're on windows, download and run the installer. It should install
a self-contained set of tools and libraries to `C:\Program Files\Rust`,
and add `C:\Program Files\Rust\bin` to your `PATH` environment variable.
You should then be able to run the rust compiler as `rustc` directly
from the command line.

We hope to be distributing binary packages for various other operating
systems at some point in the future, but at the moment only windows
binary installers are being made. Other operating systems must build
from "source".

***Note:*** The Rust compiler is slightly unusual in that it is written
in Rust and therefore must be built by a precompiled "snapshot" version
of itself (made in an earlier state of development). As such, source
builds require that:

  * You are connected to the internet, to fetch snapshots.
  * You can at least execute snapshot binaries of one of the forms we
    offer them in. Currently we build and test snapshots on:
    * Windows (7, server 2008 r2) x86 only
    * Linux 2.6.x (various distributions) x86 and x86-64
    * OSX 10.6 ("Snow leopard") or 10.7 ("Lion") x86 and x86-64

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

Assuming you're on a relatively modern Linux system and have met the
prerequisites, something along these lines should work:

~~~~
## notrust
$ wget http://dl.rust-lang.org/dist/rust-0.1.tar.gz
$ tar -xzf rust-0.1.tar.gz
$ cd rust-0.1
$ ./configure
$ make && make install
~~~~

When complete, `make install` will place the following programs into
`/usr/local/bin`:

  * `rustc`, the Rust compiler
  * `rustdoc`, the API-documentation tool 
  * `cargo`, the Rust package manager

In addition to a manual page under `/usr/local/share/man` and
a set of host and target libraries under `/usr/local/lib/rustc`.

The install locations can be adjusted by passing a `--prefix` argument
to `configure`. Various other options are also supported, pass `--help`
for more information on them. 

[installer]: http://dl.rust-lang.org/dist/rust-0.1-installer.exe
[tarball]: http://dl.rust-lang.org/dist/rust-0.1.tar.gz

## Compiling your first program

Rust program files are, by convention, given the extension `.rs`. Say
we have a file `hello.rs` containing this program:

~~~~
use std;
fn main(args: [str]) {
    std::io::println("hello world from '" + args[0] + "'!");
}
~~~~

If the Rust compiler was installed successfully, running `rustc
hello.rs` will produce a binary called `hello` (or `hello.exe`).

If you modify the program to make it invalid (for example, remove the
`use std` line), and then compile it, you'll see an error message like
this:

~~~~
## notrust
hello.rs:2:4: 2:20 error: unresolved modulename: std
hello.rs:2     std::io::println("hello world!");
               ^~~~~~~~~~~~~~~~
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

[std]: http://doc.rust-lang.org/doc/std/index/General.html

## Editing Rust code

There are Vim highlighting and indentation scripts in the Rust source
distribution under `src/etc/vim/`, and an emacs mode under
`src/etc/emacs/`.

[rust-mode]: https://github.com/marijnh/rust-mode

Other editors are not provided for yet. If you end up writing a Rust
mode for your favorite editor, let us know so that we can link to it.

# Syntax Basics

## Braces

Assuming you've programmed in any C-family language (C++, Java,
JavaScript, C#, or PHP), Rust will feel familiar. The main surface
difference to be aware of is that the bodies of `if` statements and of
loops *have* to be wrapped in brackets. Single-statement, bracket-less
bodies are not allowed.

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
of languages. A lot of thing that are statements in C are expressions
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

NOTE: The parser doesn't currently recognize non-ascii alphabetic
characters. This is a bug that will eventually be fixed.

The double-colon (`::`) is used as a module separator, so
`std::io::println` means 'the thing named `println` in the module
named `io` in the module named `std`'.

Rust will normally emit warnings about unused variables. These can be
suppressed by using a variable name that starts with an underscore.

~~~~
fn this_warns(x: int) {}
fn this_doesnt(_x: int) {}
~~~~

## Variable declaration

The `let` keyword, as we've seen, introduces a local variable. Global
constants can be defined with `const`:

~~~~
use std;
const repeat: uint = 5u;
fn main() {
    let count = 0u;
    while count < repeat {
        std::io::println("Hi!");
        count += 1u;
    }
}
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
let x = [];
# x = [3];
// Explicitly say this is a vector of integers.
let y: [int] = [];
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

`[T]`
  : Vector type.

`[mutable T]`
  : Mutable vector type.

`(T1, T2)`
  : Tuple type. Any arity above 1 is supported.

`{field1: T1, field2: T2}`
  : Record type.

`fn(arg1: T1, arg2: T2) -> T3`, `lambda()`, `block()`
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

## Literals

Integers can be written in decimal (`144`), hexadecimal (`0x90`), and
binary (`0b10010000`) base. Without a suffix, an integer literal is
considered to be of type `int`. Add a `u` (`144u`) to make it a `uint`
instead. Literals of the fixed-size integer types can be created by
the literal with the type name (`255u8`, `50i64`, etc).

Note that, in Rust, no implicit conversion between integer types
happens. If you are adding one to a variable of type `uint`, you must
type `v += 1u`—saying `+= 1` will give you a type error.

Floating point numbers are written `0.0`, `1e6`, or `2.1e-4`. Without
a suffix, the literal is assumed to be of type `float`. Suffixes `f32`
and `f64` can be used to create literals of a specific type. The
suffix `f` can be used to write `float` literals without a dot or
exponent: `3f`.

The nil literal is written just like the type: `()`. The keywords
`true` and `false` produce the boolean literals.

Character literals are written between single quotes, as in `'x'`. You
may put non-ascii characters between single quotes (your source files
should be encoded as UTF-8). Rust understands a number of
character escapes, using the backslash character:

`\n`
  : A newline (Unicode character 32).

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

Rust's set of operators contains very few surprises. The main
difference with C is that `++` and `--` are missing, and that the
logical binary operators have higher precedence—in C, `x & 2 > 0`
comes out as `x & (2 > 0)`, in Rust, it means `(x & 2) > 0`, which is
more likely to be what you expect (unless you are a C veteran).

Thus, binary arithmetic is done with `*`, `/`, `%`, `+`, and `-`
(multiply, divide, remainder, plus, minus). `-` is also a unary prefix
operator (there are no unary postfix operators in Rust) that does
negation.

Binary shifting is done with `>>` (shift right), `>>>` (arithmetic
shift right), and `<<` (shift left). Logical bitwise operators are
`&`, `|`, and `^` (and, or, and exclusive or), and unary `!` for
bitwise negation (or boolean negation when applied to a boolean
value).

The comparison operators are the traditional `==`, `!=`, `<`, `>`,
`<=`, and `>=`. Short-circuiting (lazy) boolean operators are written
`&&` (and) and `||` (or).

Rust has a ternary conditional operator `?:`, as in:

~~~~
let badness = 12;
let message = badness < 10 ? "error" : "FATAL ERROR";
~~~~

For type casting, Rust uses the binary `as` operator, which has a
precedence between the bitwise combination operators (`&`, `|`, `^`)
and the comparison operators. It takes an expression on the left side,
and a type on the right side, and will, if a meaningful conversion
exists, convert the result of the expression to the given type.

~~~~
let x: float = 4.0;
let y: uint = x as uint;
assert y == 4u;
~~~~

## Attributes

Every definition can be annotated with attributes. Attributes are meta
information that can serve a variety of purposes. One of those is
conditional compilation:

~~~~
#[cfg(target_os = "win32")]
fn register_win_service() { /* ... */ }
~~~~

This will cause the function to vanish without a trace during
compilation on a non-Windows platform, much like `#ifdef` in C (it
allows `cfg(flag=value)` and `cfg(flag)` forms, where the second
simply checks whether the configuration flag is defined at all). Flags
for `target_os` and `target_arch` are set by the compiler. It is
possible to set additional flags with the `--cfg` command-line option.

Attributes are always wrapped in hash-braces (`#[attr]`). Inside the
braces, a small minilanguage is supported, whose interpretation
depends on the attribute that's being used. The simplest form is a
plain name (as in `#[test]`, which is used by the [built-in test
framework](#testing)). A name-value pair can be provided using an `=`
character followed by a literal (as in `#[license = "BSD"]`, which is
a valid way to annotate a Rust program as being released under a
BSD-style license). Finally, you can have a name followed by a
comma-separated list of nested attributes, as in the `cfg` example
above, or in this [crate](#modules-and-crates) metadata declaration:

~~~~
## ignore
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
std::io::println(#fmt("%s is %d", "the answer", 42));
~~~~

`#fmt` supports most of the directives that [printf][pf] supports, but
will give you a compile-time error when the types of the directives
don't match the types of the arguments.

[pf]: http://en.cppreference.com/w/cpp/io/c/fprintf

All syntax extensions look like `#word`. Another built-in one is
`#env`, which will look up its argument as an environment variable at
compile-time.

~~~~
std::io::println(#env("PATH"));
~~~~
# Control structures

## Conditionals

We've seen `if` pass by a few times already. To recap, braces are
compulsory, an optional `else` clause can be appended, and multiple
`if`/`else` constructs can be chained together:

~~~~
if false {
    std::io::println("that's odd");
} else if true {
    std::io::println("right");
} else {
    std::io::println("neither true nor false");
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
  0       { std::io::println("zero"); }
  1 | 2   { std::io::println("one or two"); }
  3 to 10 { std::io::println("three to ten"); }
  _       { std::io::println("something else"); }
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

You may only use irrevocable patterns—patterns that can never fail to
match—in let bindings, though. Things like literals, which only match
a specific value, are not allowed.

## Loops

`while` produces a loop that runs as long as its given condition
(which must have type `bool`) evaluates to true. Inside a loop, the
keyword `break` can be used to abort the loop, and `cont` can be used
to abort the current iteration and continue with the next.

~~~~
let x = 5;
while true {
    x += x - 3;
    if x % 5 == 0 { break; }
    std::io::println(int::str(x));
}
~~~~

This code prints out a weird sequence of numbers and stops as soon as
it finds one that can be divided by five.

There's also `while`'s ugly cousin, `do`/`while`, which does not check
its condition on the first iteration, using traditional syntax:

~~~~
# fn eat_cake() {}
# fn any_cake_left() -> bool { false }
do {
    eat_cake();
} while any_cake_left();
~~~~

When iterating over a vector, use `for` instead.

~~~~
for elt in ["red", "green", "blue"] {
    std::io::println(elt);
}
~~~~

This will go over each element in the given vector (a three-element
vector of strings, in this case), and repeatedly execute the body with
`elt` bound to the current element. You may add an optional type
declaration (`elt: str`) for the iteration variable if you want.

For more involved iteration, such as going over the elements of a hash
table, Rust uses higher-order functions. We'll come back to those in a
moment.

## Failure

The `fail` keyword causes the current [task](#tasks) to fail. You use
it to indicate unexpected failure, much like you'd use `exit(1)` in a
C program, except that in Rust, it is possible for other tasks to
handle the failure, allowing the program to continue running.

`fail` takes an optional argument, which must have type `str`. Trying
to access a vector out of bounds, or running a pattern match with no
matching clauses, both result in the equivalent of a `fail`.

## Logging

Rust has a built-in logging mechanism, using the `log` statement.
Logging is polymorphic—any type of value can be logged, and the
runtime will do its best to output a textual representation of the
value.

~~~~
log(warn, "hi");
log(error, (1, [2.5, -1.8]));
~~~~

The first argument is the log level (levels `info`, `warn`, and
`error` are predefined), and the second is the value to log. By
default, you *will not* see the output of that first log statement,
which has `warn` level. The environment variable `RUST_LOG` controls
which log level is used. It can contain a comma-separated list of
paths for modules that should be logged. For example, running `rustc`
with `RUST_LOG=rustc::front::attr` will turn on logging in its
attribute parser. If you compile a program named `foo.rs`, its
top-level module will be called `foo`, and you can set `RUST_LOG` to
`foo` to enable `warn` and `info` logging for the module.

Turned-off `log` statements impose minimal overhead on the code that
contains them, so except in code that needs to be really, really fast,
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

## Assertions

The keyword `assert`, followed by an expression with boolean type,
will check that the given expression results in `true`, and cause a
failure otherwise. It is typically used to double-check things that
*should* hold at a certain point in a program.

~~~~
let x = 100;
while (x > 10) { x -= 10; }
assert x == 10;
~~~~

# Functions

Functions (like all other static declarations, such as `type`) can be
declared both at the top level and inside other functions (or modules,
which we'll come back to in moment).

The `ret` keyword immediately returns from a function. It is
optionally followed by an expression to return. In functions that
return `()`, the returned expression can be left off. A function can
also return a value by having its top level block produce an
expression (by omitting the final semicolon).

Some functions (such as the C function `exit`) never return normally.
In Rust, these are annotated with the pseudo-return type '`!`':

~~~~
fn dead_end() -> ! { fail; }
~~~~

This helps the compiler avoid spurious error messages. For example,
the following code would be a type error if `dead_end` would be
expected to return.

~~~~
# fn can_go_left() -> bool { true }
# fn can_go_right() -> bool { true }
# enum dir { left; right; }
# fn dead_end() -> ! { fail; }
let dir = if can_go_left() { left }
          else if can_go_right() { right }
          else { dead_end(); };
~~~~

## Closures

Named functions, like those in the previous section, do not close over
their environment. Rust also includes support for closures, which are
functions that can access variables in the scope in which they are
created.

There are several forms of closures, each with its own role. The most
common type is called a 'block', this is a closure which has full
access to its environment.

~~~~
fn call_block_with_ten(b: block(int)) { b(10); }

let x = 20;    
call_block_with_ten({|arg|
    #info("x=%d, arg=%d", x, arg);
});
~~~~

This defines a function that accepts a block, and then calls it with a
simple block that executes a log statement, accessing both its
argument and the variable `x` from its environment.

Blocks can only be used in a restricted way, because it is not allowed
to survive the scope in which it was created. They are allowed to
appear in function argument position and in call position, but nowhere
else.

### Boxed closures

When you need to store a closure in a data structure, a block will not
do, since the compiler will refuse to let you store it. For this
purpose, Rust provides a type of closure that has an arbitrary
lifetime, written `fn@` (boxed closure, analogous to the `@` pointer
type described in the next section).

A boxed closure does not directly access its environment, but merely
copies out the values that it closes over into a private data
structure. This means that it can not assign to these variables, and
will not 'see' updates to them.

This code creates a closure that adds a given string to its argument,
returns it from a function, and then calls it:

~~~~
use std;

fn mk_appender(suffix: str) -> fn@(str) -> str {
    let f = fn@(s: str) -> str { s + suffix };
    ret f;
}

fn main() {
    let shout = mk_appender("!");
    std::io::println(shout("hey ho, let's go"));
}
~~~~

### Closure compatibility

A nice property of Rust closures is that you can pass any kind of
closure (as long as the arguments and return types match) to functions
that expect a `block`. Thus, when writing a higher-order function that
wants to do nothing with its function argument beyond calling it, you
should almost always specify the type of that argument as `block`, so
that callers have the flexibility to pass whatever they want.

~~~~
fn call_twice(f: block()) { f(); f(); }
call_twice({|| "I am a block"; });
call_twice(fn@() { "I am a boxed closure"; });
fn bare_function() { "I am a plain function"; }
call_twice(bare_function);
~~~~

### Unique closures

Unique closures, written `fn~` in analogy to the `~` pointer type (see
next section), hold on to things that can safely be sent between
processes. They copy the values they close over, much like boxed
closures, but they also 'own' them—meaning no other code can access
them. Unique closures mostly exist to for spawning new
[tasks](#tasks).

### Shorthand syntax

The compact syntax used for blocks (`{|arg1, arg2| body}`) can also
be used to express boxed and unique closures in situations where the
closure style can be unambiguously derived from the context. Most
notably, when calling a higher-order function you do not have to use
the long-hand syntax for the function you're passing, since the
compiler can look at the argument type to find out what the parameter
types are.

As a further simplification, if the final parameter to a function is a
closure, the closure need not be placed within parentheses. You could,
for example, write...

~~~~
let doubled = vec::map([1, 2, 3]) {|x| x*2};
~~~~

`vec::map` is a function in the core library that applies its last
argument to every element of a vector, producing a new vector.

Even when a closure takes no parameters, you must still write the bars
for the parameter list, as in `{|| ...}`.

## Binding

Partial application is done using the `bind` keyword in Rust.

~~~~
let daynum = bind vec::position(_, ["mo", "tu", "we", "do",
                                    "fr", "sa", "su"]);
~~~~

Binding a function produces a boxed closure (`fn@` type) in which some
of the arguments to the bound function have already been provided.
`daynum` will be a function taking a single string argument, and
returning the day of the week that string corresponds to (if any).

## Iteration

Functions taking blocks provide a good way to define non-trivial
iteration constructs. For example, this one iterates over a vector
of integers backwards:

~~~~
fn for_rev(v: [int], act: block(int)) {
    let i = vec::len(v);
    while (i > 0u) {
        i -= 1u;
        act(v[i]);
    }
}
~~~~

To run such an iteration, you could do this:

~~~~
# fn for_rev(v: [int], act: block(int)) {}
for_rev([1, 2, 3], {|n| log(error, n); });
~~~~

Making use of the shorthand where a final closure argument can be
moved outside of the parentheses permits the following, which
looks quite like a normal loop:

~~~~
# fn for_rev(v: [int], act: block(int)) {}
for_rev([1, 2, 3]) {|n|
    log(error, n);
}
~~~~

Note that, because `for_rev()` returns unit type, no semicolon is
needed when the final closure is pulled outside of the parentheses.

# Datatypes

Rust datatypes are, by default, immutable. The core datatypes of Rust
are structural records and 'enums' (tagged unions, algebraic data
types).

~~~~
type point = {x: float, y: float};
enum shape {
    circle(point, float);
    rectangle(point, point);
}
let my_shape = circle({x: 0.0, y: 0.0}, 10.0);
~~~~

## Records

Rust record types are written `{field1: TYPE, field2: TYPE [, ...]}`,
and record literals are written in the same way, but with expressions
instead of types. They are quite similar to C structs, and even laid
out the same way in memory (so you can read from a Rust struct in C,
and vice-versa).

The dot operator is used to access record fields (`mypoint.x`).

Fields that you want to mutate must be explicitly marked as such. For
example...

~~~~
type stack = {content: [int], mutable head: uint};
~~~~

With such a type, you can do `mystack.head += 1u`. If `mutable` were
omitted from the type, such an assignment would result in a type
error.

To 'update' an immutable record, you use functional record update
syntax, by ending a record literal with the keyword `with`:

~~~~
let oldpoint = {x: 10f, y: 20f};
let newpoint = {x: 0f with oldpoint};
assert newpoint == {x: 0f, y: 20f};
~~~~

This will create a new struct, copying all the fields from `oldpoint`
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

Records can be destructured on in `alt` patterns. The basic syntax is
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

Enums are datatypes that have several different representations. For
example, the type shown earlier:

~~~~
# type point = {x: float, y: float};
enum shape {
    circle(point, float);
    rectangle(point, point);
}
~~~~

A value of this type is either a circle¸ in which case it contains a
point record and a float, or a rectangle, in which case it contains
two point records. The run-time representation of such a value
includes an identifier of the actual form that it holds, much like the
'tagged union' pattern in C, but with better ergonomics.

The above declaration will define a type `shape` that can be used to
refer to such shapes, and two functions, `circle` and `rectangle`,
which can be used to construct values of the type (taking arguments of
the specified types). So `circle({x: 0f, y: 0f}, 10f)` is the way to
create a new circle.

Enum variants do not have to have parameters. This, for example, is
equivalent to a C enum:

~~~~
enum direction {
    north;
    east;
    south;
    west;
}
~~~~

This will define `north`, `east`, `south`, and `west` as constants,
all of which have type `direction`.

When the enum is C like, that is none of the variants have parameters,
it is possible to explicitly set the discriminator values to an integer
value:

~~~~
enum color {
  red = 0xff0000;
  green = 0x00ff00;
  blue = 0x0000ff;
}
~~~~

If an explicit discriminator is not specified for a variant, the value
defaults to the value of the previous variant plus one.  If the first
variant does not have a discriminator, it defaults to 0.  For example,
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
enum gizmo_id { gizmo_id(int); }
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
# enum shape { circle(point, float); rectangle(point, point); }
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
# enum direction { north; east; south; west; }
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
Tuples can have any arity except for 0 or 1 (though you may see nil,
`()`, as the empty tuple if you like).

~~~~
let mytup: (int, int, float) = (10, 20, 30.0);
alt mytup {
  (a, b, c) { log(info, a + b + (c as int)); }
}
~~~~

## Pointers

In contrast to a lot of modern languages, record and enum types in
Rust are not represented as pointers to allocated memory. They are,
like in C and C++, represented directly. This means that if you `let x
= {x: 1f, y: 1f};`, you are creating a record on the stack. If you
then copy it into a data structure, the whole record is copied, not
just a pointer.

For small records like `point`, this is usually more efficient than
allocating memory and going through a pointer. But for big records, or
records with mutable fields, it can be useful to have a single copy on
the heap, and refer to that through a pointer.

Rust supports several types of pointers. The simplest is the unsafe
pointer, written `*TYPE`, which is a completely unchecked pointer
type only used in unsafe code (and thus, in typical Rust code, very
rarely). The safe pointer types are `@TYPE` for shared,
reference-counted boxes, and `~TYPE`, for uniquely-owned pointers.

All pointer types can be dereferenced with the `*` unary operator.

### Shared boxes

Shared boxes are pointers to heap-allocated, reference counted memory.
A cycle collector ensures that circular references do not result in
memory leaks.

Creating a shared box is done by simply applying the unary `@`
operator to an expression. The result of the expression will be boxed,
resulting in a box of the right type. For example:

~~~~
let x = @10; // New box, refcount of 1
let y = x; // Copy the pointer, increase refcount
// When x and y go out of scope, refcount goes to 0, box is freed
~~~~

NOTE: We may in the future switch to garbage collection, rather than
reference counting, for shared boxes.

Shared boxes never cross task boundaries.

### Unique boxes

In contrast to shared boxes, unique boxes are not reference counted.
Instead, it is statically guaranteed that only a single owner of the
box exists at any time.

~~~~
let x = ~10;
let y <- x;
~~~~

This is where the 'move' (`<-`) operator comes in. It is similar to
`=`, but it de-initializes its source. Thus, the unique box can move
from `x` to `y`, without violating the constraint that it only has a
single owner (if you used assignment instead of the move operator, the
box would, in principle, be copied).

Unique boxes, when they do not contain any shared boxes, can be sent
to other tasks. The sending task will give up ownership of the box,
and won't be able to access it afterwards. The receiving task will
become the sole owner of the box.

### Mutability

All pointer types have a mutable variant, written `@mutable TYPE` or
`~mutable TYPE`. Given such a pointer, you can write to its contents
by combining the dereference operator with a mutating action.

~~~~
fn increase_contents(pt: @mutable int) {
    *pt += 1;
}
~~~~

## Vectors

Rust vectors are always heap-allocated and unique. A value of type
`[TYPE]` is represented by a pointer to a section of heap memory
containing any number of `TYPE` values.

NOTE: This uniqueness is turning out to be quite awkward in practice,
and might change in the future.

Vector literals are enclosed in square brackets. Dereferencing is done
with square brackets (zero-based):

~~~~
let myvec = [true, false, true, false];
if myvec[1] { std::io::println("boom"); }
~~~~

By default, vectors are immutable—you can not replace their elements.
The type written as `[mutable TYPE]` is a vector with mutable
elements. Mutable vector literals are written `[mutable]` (empty) or
`[mutable 1, 2, 3]` (with elements).

The `+` operator means concatenation when applied to vector types.
Growing a vector in Rust is not as inefficient as it looks :

~~~~
let myvec = [], i = 0;
while i < 100 {
    myvec += [i];
    i += 1;
}
~~~~

Because a vector is unique, replacing it with a longer one (which is
what `+= [i]` does) is indistinguishable from appending to it
in-place. Vector representations are optimized to grow
logarithmically, so the above code generates about the same amount of
copying and reallocation as `push` implementations in most other
languages.

## Strings

The `str` type in Rust is represented exactly the same way as a vector
of bytes (`[u8]`), except that it is guaranteed to have a trailing
null byte (for interoperability with C APIs).

This sequence of bytes is interpreted as an UTF-8 encoded sequence of
characters. This has the advantage that UTF-8 encoded I/O (which
should really be the default for modern systems) is very fast, and
that strings have, for most intents and purposes, a nicely compact
representation. It has the disadvantage that you only get
constant-time access by byte, not by character.

A lot of algorithms don't need constant-time indexed access (they
iterate over all characters, which `str::chars` helps with), and
for those that do, many don't need actual characters, and can operate
on bytes. For algorithms that do really need to index by character,
there's the option to convert your string to a character vector (using
`str::to_chars`).

Like vectors, strings are always unique. You can wrap them in a shared
box to share them. Unlike vectors, there is no mutable variant of
strings. They are always immutable.

## Resources

Resources are data types that have a destructor associated with them.

~~~~
# fn close_file_desc(x: int) {}
resource file_desc(fd: int) {
    close_file_desc(fd);
}
~~~~

This defines a type `file_desc` and a constructor of the same name,
which takes an integer. Values of such a type can not be copied, and
when they are destroyed (by going out of scope, or, when boxed, when
their box is cleaned up), their body runs. In the example above, this
would cause the given file descriptor to be closed.

NOTE: We're considering alternative approaches for data types with
destructors. Resources might go away in the future.

# Argument passing

Rust datatypes are not trivial to copy (the way, for example,
JavaScript values can be copied by simply taking one or two machine
words and plunking them somewhere else). Shared boxes require
reference count updates, big records, tags, or unique pointers require
an arbitrary amount of data to be copied (plus updating the reference
counts of shared boxes hanging off them).

For this reason, the default calling convention for Rust functions
leaves ownership of the arguments with the caller. The caller
guarantees that the arguments will outlive the call, the callee merely
gets access to them.

## Safe references

There is one catch with this approach: sometimes the compiler can
*not* statically guarantee that the argument value at the caller side
will survive to the end of the call. Another argument might indirectly
refer to it and be used to overwrite it, or a closure might assign a
new value to it.

Fortunately, Rust tasks are single-threaded worlds, which share no
data with other tasks, and that most data is immutable. This allows
most argument-passing situations to be proved safe without further
difficulty.

Take the following program:

~~~~
# fn get_really_big_record() -> int { 1 }
# fn myfunc(a: int) {}
fn main() {
    let x = get_really_big_record();
    myfunc(x);
}
~~~~

Here we know for sure that no one else has access to the `x` variable
in `main`, so we're good. But the call could also look like this:

~~~~
# fn myfunc(a: int, b: block()) {}
# fn get_another_record() -> int { 1 }
# let x = 1;
myfunc(x, {|| x = get_another_record(); });
~~~~

Now, if `myfunc` first calls its second argument and then accesses its
first argument, it will see a different value from the one that was
passed to it.

In such a case, the compiler will insert an implicit copy of `x`,
*except* if `x` contains something mutable, in which case a copy would
result in code that behaves differently. If copying `x` might be
expensive (for example, if it holds a vector), the compiler will emit
a warning.

There are even more tricky cases, in which the Rust compiler is forced
to pessimistically assume a value will get mutated, even though it is
not sure.

~~~~
fn for_each(v: [mutable @int], iter: block(@int)) {
   for elt in v { iter(elt); }
}
~~~~

For all this function knows, calling `iter` (which is a closure that
might have access to the vector that's passed as `v`) could cause the
elements in the vector to be mutated, with the effect that it can not
guarantee that the boxes will live for the duration of the call. So it
has to copy them. In this case, this will happen implicitly (bumping a
reference count is considered cheap enough to not warn about it).

## The copy operator

If the `for_each` function given above were to take a vector of
`{mutable a: int}` instead of `@int`, it would not be able to
implicitly copy, since if the `iter` function changes a copy of a
mutable record, the changes won't be visible in the record itself. If
we *do* want to allow copies there, we have to explicitly allow it
with the `copy` operator:

~~~~
type mutrec = {mutable x: int};
fn for_each(v: [mutable mutrec], iter: block(mutrec)) {
   for elt in v { iter(copy elt); }
}
~~~~

Adding a `copy` operator is also the way to muffle warnings about
implicit copies.

## Other uses of safe references

Safe references are not only used for argument passing. When you
destructure on a value in an `alt` expression, or loop over a vector
with `for`, variables bound to the inside of the given data structure
will use safe references, not copies. This means such references are
very cheap, but you'll occasionally have to copy them to ensure
safety.

~~~~
let my_rec = {a: 4, b: [1, 2, 3]};
alt my_rec {
  {a, b} {
    log(info, b); // This is okay
    my_rec = {a: a + 1, b: b + [a]};
    log(info, b); // Here reference b has become invalid
  }
}
~~~~

## Argument passing styles

The fact that arguments are conceptually passed by safe reference does
not mean all arguments are passed by pointer. Composite types like
records and tags *are* passed by pointer, but single-word values, like
integers and pointers, are simply passed by value. Most of the time,
the programmer does not have to worry about this, as the compiler will
simply pick the most efficient passing style. There is one exception,
which will be described in the section on [generics](#generics).

To explicitly set the passing-style for a parameter, you prefix the
argument name with a sigil. There are two special passing styles that
are often useful. The first is by-mutable-pointer, written with a
single `&`:

~~~~
fn vec_push(&v: [int], elt: int) {
    v += [elt];
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

# Generics

## Generic functions

Throughout this tutorial, I've been defining functions like `for_rev`
that act only on integers. It is 2012, and we no longer expect to be
defining such functions again and again for every type they apply to.
Thus, Rust allows functions and datatypes to have type parameters.

~~~~
fn for_rev<T>(v: [T], act: block(T)) {
    let i = vec::len(v);
    while i > 0u {
        i -= 1u;
        act(v[i]);
    }
}

fn map<T, U>(v: [T], f: block(T) -> U) -> [U] {
    let acc = [];
    for elt in v { acc += [f(elt)]; }
    ret acc;
}
~~~~

When defined in this way, these functions can be applied to any type
of vector, as long as the type of the block's argument and the type of
the vector's content agree with each other.

Inside a parameterized (generic) function, the names of the type
parameters (capitalized by convention) stand for opaque types. You
can't look inside them, but you can pass them around.

## Generic datatypes

Generic `type` and `enum` declarations follow the same pattern:

~~~~
type circular_buf<T> = {start: uint,
                        end: uint,
                        buf: [mutable T]};

enum option<T> { some(T); none; }
~~~~

You can then declare a function to take a `circular_buf<u8>` or return
an `option<str>`, or even an `option<T>` if the function itself is
generic.

The `option` type given above exists in the core library as
`option::t`, and is the way Rust programs express the thing that in C
would be a nullable pointer. The nice part is that you have to
explicitly unpack an `option` type, so accidental null pointer
dereferences become impossible.

## Type-inference and generics

Rust's type inferrer works very well with generics, but there are
programs that just can't be typed.

~~~~
let n = option::none;
# n = option::some(1);
~~~~

If you never do anything else with `n`, the compiler will not be able
to assign a type to it. (The same goes for `[]`, the empty vector.) If
you really want to have such a statement, you'll have to write it like
this:

~~~~
let n2: option::t<int> = option::none;
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
for all Rust types. Resource types (types with destructors) can not be
copied, and neither can any type whose copying would require copying a
resource (such as records or unique boxes containing a resource).

This complicates handling of generic functions. If you have a type
parameter `T`, can you copy values of that type? In Rust, you can't,
unless you explicitly declare that type parameter to have copyable
'kind'. A kind is a type of type.

~~~~
## ignore
// This does not compile
fn head_bad<T>(v: [T]) -> T { v[0] }
// This does
fn head<T: copy>(v: [T]) -> T { v[0] }
~~~~

When instantiating a generic function, you can only instantiate it
with types that fit its kinds. So you could not apply `head` to a
resource type.

Rust has three kinds: 'noncopyable', 'copyable', and 'sendable'. By
default, type parameters are considered to be noncopyable. You can
annotate them with the `copy` keyword to declare them copyable, and
with the `send` keyword to make them sendable.

Sendable types are a subset of copyable types. They are types that do
not contain shared (reference counted) types, which are thus uniquely
owned by the function that owns them, and can be sent over channels to
other tasks. Most of the generic functions in the core `comm` module
take sendable types.

## Generic functions and argument-passing

The previous section mentioned that arguments are passed by pointer or
by value based on their type. There is one situation in which this is
difficult. If you try this program:

~~~~
# fn map(f: block(int) -> int, v: [int]) {}
fn plus1(x: int) -> int { x + 1 }
map(plus1, [1, 2, 3]);
~~~~

You will get an error message about argument passing styles
disagreeing. The reason is that generic types are always passed by
pointer, so `map` expects a function that takes its argument by
pointer. The `plus1` you defined, however, uses the default, efficient
way to pass integers, which is by value. To get around this issue, you
have to explicitly mark the arguments to a function that you want to
pass to a generic higher-order function as being passed by pointer,
using the `&&` sigil:

~~~~
# fn map<T, U>(f: block(T) -> U, v: [T]) {}
fn plus1(&&x: int) -> int { x + 1 }
map(plus1, [1, 2, 3]);
~~~~

NOTE: This is inconvenient, and we are hoping to get rid of this
restriction in the future.

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
    std::io::println(farm::chicken());
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

~~~~
## ignore
#[link(name = "farm", vers = "2.5", author = "mjh")];
mod cow;
mod chicken;
mod horse;
~~~~

Compiling this file will cause `rustc` to look for files named
`cow.rs`, `chicken.rs`, `horse.rs` in the same directory as the `.rc`
file, compile them all together, and, depending on the presence of the
`--lib` switch, output a shared library or an executable.

The `#[link(...)]` part provides meta information about the module,
which other crates can use to load the right module. More about that
later.

To have a nested directory structure for your source files, you can
nest mods in your `.rc` file:

~~~~
## ignore
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

Having compiled a crate with `--lib`, you can use it in another crate
with a `use` directive. We've already seen `use std` in several of the
examples, which loads in the [standard library][std].

[std]: http://doc.rust-lang.org/doc/std/index/General.html

`use` directives can appear in a crate file, or at the top level of a
single-file `.rs` crate. They will cause the compiler to search its
library search path (which you can extend with `-L` switch) for a Rust
crate library with the right name.

It is possible to provide more specific information when using an
external crate.

~~~~
## ignore
use myfarm (name = "farm", vers = "2.7");
~~~~

When a comma-separated list of name/value pairs is given after `use`,
these are matched against the attributes provided in the `link`
attribute of the crate file, and a crate is only used when the two
match. A `name` value can be given to override the name used to search
for the crate. So the above would import the `farm` crate under the
local name `myfarm`.

Our example crate declared this set of `link` attributes:

~~~~
## ignore
#[link(name = "farm", vers = "2.5", author = "mjh")];
~~~~

The version does not match the one provided in the `use` directive, so
unless the compiler can find another crate with the right version
somewhere, it will complain that no matching crate was found.

## The core library

A set of basic library routines, mostly related to built-in datatypes
and the task system, are always implicitly linked and included in any
Rust program, unless the `--no-core` compiler switch is given.

This library is documented [here][core].

[core]: http://doc.rust-lang.org/doc/core/index/General.html

## A minimal example

Now for something that you can actually compile yourself. We have
these two files:

~~~~
// mylib.rs
#[link(name = "mylib", vers = "1.0")];
fn world() -> str { "world" }
~~~~

~~~~
## ignore
// main.rs
use mylib;
fn main() { std::io::println("hello " + mylib::world()); }
~~~~

Now compile and run like this (adjust to your platform if necessary):

~~~~
## notrust
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
import std::io::println;
fn main() {
    println("that was easy");
}
~~~~

It is also possible to import just the name of a module (`import
std::io;`, then use `io::println`), to import all identifiers exported
by a given module (`import std::io::*`), or to import a specific set
of identifiers (`import math::{min, max, pi}`).

You can rename an identifier when importing using the `=` operator:

~~~~
import prnt = std::io::println;
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

Rust uses three different namespaces. One for modules, one for types,
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
that's part of a bigger crate will have that crate's context as parent
context.

Identifiers can shadow each others. In this program, `x` is of type
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

NOTE: This feature is very new, and will need a few extensions to be
applicable to more advanced use cases.

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
fn comma_sep<T: to_str>(elts: [T]) -> str {
    let result = "", first = true;
    for elt in elts {
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
    fn iter(block(T));
}
impl <T> of seq<T> for [T] {
    fn len() -> uint { vec::len(self) }
    fn iter(b: block(T)) {
        for elt in self { b(elt); }
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
# iface drawable { fn draw(); }
fn draw_all<T: drawable>(shapes: [T]) {
    for shape in shapes { shape.draw(); }
}
~~~~

You can call that on an array of circles, or an array of squares
(assuming those have suitable `drawable` interfaces defined), but not
on an array containing both circles and squares.

When this is needed, an interface name can be used as a type, causing
the function to be written simply like this:

~~~~
# iface drawable { fn draw(); }
fn draw_all(shapes: [drawable]) {
    for shape in shapes { shape.draw(); }
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
# fn draw_all(shapes: [drawable]) {}
let c: circle = new_circle();
let r: rectangle = new_rectangle();
draw_all([c as drawable, r as drawable]);
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
    fn times(b: block(int)) {
        let i = 0;
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

~~~~
use std;

native mod crypto {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}

fn as_hex(data: [u8]) -> str {
    let acc = "";
    for byte in data { acc += #fmt("%02x", byte as uint); }
    ret acc;
}

fn sha1(data: str) -> str unsafe {
    let bytes = str::bytes(data);
    let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                            vec::len(bytes), ptr::null());
    ret as_hex(vec::unsafe::from_buf(hash, 20u));
}

fn main(args: [str]) {
    std::io::println(sha1(args[1]));
}
~~~~

## Native modules

Before we can call `SHA1`, we have to declare it. That is what this
part of the program is responsible for:

~~~~
native mod crypto {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}
~~~~

A `native` module declaration tells the compiler that the program
should be linked with a library by that name, and that the given list
of functions are available in that library.

In this case, it'll change the name `crypto` to a shared library name
in a platform-specific way (`libcrypto.so` on Linux, for example), and
link that in. If you want the module to have a different name from the
actual library, you can use the `"link_name"` attribute, like:

~~~~
#[link_name = "crypto"]
native mod something {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}
~~~~

## Native calling conventions

Most native C code use the cdecl calling convention, so that is what
Rust uses by default when calling native functions. Some native functions,
most notably the Windows API, use other calling conventions, so Rust
provides a way to to hint to the compiler which is expected by using
the `"abi"` attribute:

~~~~
#[cfg(target_os = "win32")]
#[abi = "stdcall"]
native mod kernel32 {
    fn SetEnvironmentVariableA(n: *u8, v: *u8) -> int;
}
~~~~

The `"abi"` attribute applies to a native mod (it can not be applied
to a single function within a module), and must be either `"cdecl"`
or `"stdcall"`. Other conventions may be defined in the future.

## Unsafe pointers

The native `SHA1` function is declared to take three arguments, and
return a pointer.

~~~~
# native mod crypto {
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
# fn as_hex(data: [u8]) -> str { "hi" }
fn sha1(data: str) -> str unsafe {
    let bytes = str::bytes(data);
    let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                            vec::len(bytes), ptr::null());
    ret as_hex(vec::unsafe::from_buf(hash, 20u));
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
# fn as_hex(data: [u8]) -> str { "hi" }
# fn x(data: str) -> str unsafe {
let bytes = str::bytes(data);
let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                        vec::len(bytes), ptr::null());
ret as_hex(vec::unsafe::from_buf(hash, 20u));
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
type timeval = {mutable tv_sec: u32,
                mutable tv_usec: u32};
#[nolink]
native mod libc {
    fn gettimeofday(tv: *timeval, tz: *()) -> i32;
}
fn unix_time_in_microseconds() -> u64 unsafe {
    let x = {mutable tv_sec: 0u32, mutable tv_usec: 0u32};
    libc::gettimeofday(ptr::addr_of(x), ptr::null());
    ret (x.tv_sec as u64) * 1000_000_u64 + (x.tv_usec as u64);
}
~~~~

The `#[nolink]` attribute indicates that there's no native library to link
in. The standard C library is already linked with Rust programs.

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
copying it by making use of [unique boxes](#unique-boxes), which allow
the sending task to release ownership of a value, so that the
receiving task can keep on using it.

NOTE: As Rust evolves, we expect the Task API to grow and change
somewhat.  The tutorial documents the API as it exists today.

## Spawning a task

Spawning a task is done using the various spawn functions in the
module `task`.  Let's begin with the simplest one, `task::spawn()`:

~~~~
let some_value = 22;
let child_task = task::spawn {||
    std::io::println("This executes in the child task.");
    std::io::println(#fmt("%d", some_value));
};
~~~~

The argument to `task::spawn()` is a [unique
closure](#unique-closures) of type `fn~()`, meaning that it takes no
arguments and generates no return value. The effect of `task::spawn()`
is to fire up a child task that will execute the closure in parallel
with the creator. The result is a task id, here stored into the
variable `child_task`.

## Ports and channels

Now that we have spawned a child task, it would be nice if we could
communicate with it.  This is done by creating a *port* with an
associated *channel*.  A port is simply a location to receive messages
of a particular type.  A channel is used to send messages to a port.
For example, imagine we wish to perform two expensive computations
in parallel.  We might write something like:

~~~~
# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
let port = comm::port::<int>();
let chan = comm::chan::<int>(port);
let child_task = task::spawn {||
    let result = some_expensive_computation();
    comm::send(chan, result);
};
some_other_expensive_computation();
let result = comm::recv(port);
~~~~

Let's walk through this code line-by-line.  The first line creates a
port for receiving integers:

~~~~
let port = comm::port::<int>();

~~~~
This port is where we will receive the message from the child task
once it is complete.  The second line creates a channel for sending
integers to the port `port`:

~~~~
# let port = comm::port::<int>();
let chan = comm::chan::<int>(port);
~~~~

The channel will be used by the child to send a message to the port.
The next statement actually spawns the child:

~~~~
# fn some_expensive_computation() -> int { 42 }
# let port = comm::port::<int>();
# let chan = comm::chan::<int>(port);
let child_task = task::spawn {||
    let result = some_expensive_computation();
    comm::send(chan, result);
};
~~~~

This child will perform the expensive computation send the result
over the channel.  Finally, the parent continues by performing
some other expensive computation and then waiting for the child's result
to arrive on the port:

~~~~
# fn some_other_expensive_computation() {}
# let port = comm::port::<int>();
some_other_expensive_computation();
let result = comm::recv(port);
~~~~

## Creating a task with a bi-directional communication path

A very common thing to do is to spawn a child task where the parent
and child both need to exchange messages with each other. The function
`task::spawn_connected()` supports this pattern. We'll look briefly at
how it is used.

To see how `spawn_connected()` works, we will create a child task
which receives `uint` messages, converts them to a string, and sends
the string in response.  The child terminates when `0` is received.
Here is the function which implements the child task:

~~~~
fn stringifier(from_par: comm::port<uint>,
               to_par: comm::chan<str>) {
    let value: uint;
    do {
        value = comm::recv(from_par);
        comm::send(to_par, uint::to_str(value, 10u));
    } while value != 0u;
}

~~~~
You can see that the function takes two parameters.  The first is a
port used to receive messages from the parent, and the second is a
channel used to send messages to the parent.  The body itself simply
loops, reading from the `from_par` port and then sending its response
to the `to_par` channel.  The actual response itself is simply the
strified version of the received value, `uint::to_str(value)`.

Here is the code for the parent task:
~~~~

# fn stringifier(from_par: comm::port<uint>,
#                to_par: comm::chan<str>) {}
fn main() {
    let t = task::spawn_connected(stringifier);
    comm::send(t.to_child, 22u);
    assert comm::recv(t.from_child) == "22";
    comm::send(t.to_child, 23u);
    assert comm::recv(t.from_child) == "23";
    comm::send(t.to_child, 0u);
    assert comm::recv(t.from_child) == "0";
}
~~~~

The call to `spawn_connected()` on the first line will instantiate the
various ports and channels and startup the child task.  The returned
value, `t`, is a record of type `task::connected_task<uint,str>`.  In
addition to the task id of the child, this record defines two fields,
`from_child` and `to_child`, which contain the port and channel
respectively for communicating with the child.  Those fields are used
here to send and receive three messages from the child task.

## Joining a task

The function `spawn_joinable()` is used to spawn a task that can later
be joined. This is implemented by having the child task send a message
when it has completed (either successfully or by failing). Therefore,
`spawn_joinable()` returns a structure containing both the task ID and
the port where this message will be sent---this structure type is
called `task::joinable_task`. The structure can be passed to
`task::join()`, which simply blocks on the port, waiting to receive
the message from the child task.

## The supervisor relationship

By default, failures in Rust propagate upward through the task tree.
We say that each task is supervised by its parent, meaning that if the
task fails, that failure is propagated to the parent task, which will
fail sometime later.  This propagation can be disabled by using the
function `task::unsupervise()`, which disables error propagation from
the current task to its parent.

# Testing

The Rust language has a facility for testing built into the language.
Tests can be interspersed with other code, and annotated with the
`#[test]` attribute.

~~~~
use std;

fn twice(x: int) -> int { x + x }

#[test]
fn test_twice() {
    let i = -100;
    while i < 100 {
        assert twice(i) == 2 * i;
        i += 1;
    }
}
~~~~

When you compile the program normally, the `test_twice` function will
not be included. To compile and run such tests, compile with the
`--test` flag, and then run the result:

~~~~
## notrust
> rustc --test twice.rs
> ./twice
running 1 tests
test test_twice ... ok
result: ok. 1 passed; 0 failed; 0 ignored
~~~~

Or, if we change the file to fail, for example by replacing `x + x`
with `x + 1`:

~~~~
## notrust
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
~~~~

To disable a test completely, add an `#[ignore]` attribute. Running a
test runner (the program compiled with `--test`) with an `--ignored`
command-line flag will cause it to also run the tests labelled as
ignored.

A program compiled as a test runner will have the configuration flag
`test` defined, so that you can add code that won't be included in a
normal compile with the `#[cfg(test)]` attribute (see [conditional
compilation](#attributes)).
