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
large, high-performance applications while preventing several classes
of errors commonly found in languages like C++. Rust has a
sophisticated memory model that makes possible many of the efficient
data structures used in C++, while disallowing invalid memory accesses
that would otherwise cause segmentation faults. Like other systems
languages, it is statically typed and compiled ahead of time.

As a multi-paradigm language, Rust supports writing code in
procedural, functional and object-oriented styles. Some of its nice
high-level features include:

* ***Pattern matching and algebraic data types (enums).*** Common in
  functional languages, pattern matching on ADTs provides a compact
  and expressive way to encode program logic.
* ***Task-based concurrency.*** Rust uses lightweight tasks that do
  not share memory.
* ***Higher-order functions.*** Rust functions may take closures as
  arguments or return closures as return values.  Closures in Rust are
  very powerful and used pervasively.
* ***Trait polymorphism.*** Rust's type system features a unique
  combination of Java-style interfaces and Haskell-style typeclasses
  called _traits_.
* ***Parametric polymorphism (generics).*** Functions and types can be
  parameterized over type variables with optional type constraints.
* ***Type inference.*** Type annotations on local variable
  declarations can be omitted.

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
    return result;
}
~~~~

Several differences from C stand out. Types do not come before, but
after variable names (preceded by a colon). For local variables
(introduced with `let`), types are optional, and will be inferred when
left off. Constructs like `while` and `if` do not require parentheses
around the condition (though they allow them).

You should, however, not conclude that Rust is simply an evolution of
C. As will become clear in the rest of this tutorial, it goes in quite
a different direction, with efficient, strongly-typed and memory-safe
support for many high-level idioms.

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
$ wget http://dl.rust-lang.org/dist/rust-0.3.tar.gz
$ tar -xzf rust-0.3.tar.gz
$ cd rust-0.3
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
[tarball]: http://dl.rust-lang.org/dist/rust-0.3.tar.gz

## Compiling your first program

Rust program files are, by convention, given the extension `.rs`. Say
we have a file `hello.rs` containing this program:

~~~~
fn main() {
    io::println("hello world!");
}
~~~~

If the Rust compiler was installed successfully, running `rustc
hello.rs` will produce a binary called `hello` (or `hello.exe`).

If you modify the program to make it invalid (for example, by changing
 `io::println` to some nonexistent function), and then compile it,
 you'll see an error message like this:

~~~~ {.notrust}
hello.rs:2:4: 2:16 error: unresolved name: io::print_it
hello.rs:2     io::print_it("hello world!");
               ^~~~~~~~~~~~
~~~~

The Rust compiler tries to provide useful information when it runs
into an error.

## Anatomy of a Rust program

In its simplest form, a Rust program is a `.rs` file with some
types and functions defined in it. If it has a `main` function, it can
be compiled to an executable. Rust does not allow code that's not a
declaration to appear at the top level of the file—all statements must
live inside a function.

Rust programs can also be compiled as libraries, and included in other
programs. The `extern mod std` directive that appears at the top of a lot of
examples imports the [standard library][std]. This is described in more
detail [later on](#modules-and-crates).

[std]: http://doc.rust-lang.org/doc/std

## Editing Rust code

There are Vim highlighting and indentation scripts in the Rust source
distribution under `src/etc/vim/`, and an emacs mode under
`src/etc/emacs/`. There is a package for Sublime Text 2 at
[github.com/dbp/sublime-rust](http://github.com/dbp/sublime-rust), also
available through [package control](http://wbond.net/sublime_packages/package_control).

Other editors are not provided for yet. If you end up writing a Rust
mode for your favorite editor, let us know so that we can link to it.

# Syntax Basics

## Braces

Assuming you've programmed in any C-family language (C++, Java,
JavaScript, C#, or PHP), Rust will feel familiar. The main surface
difference to be aware of is that the bodies of `if` statements and of
`while` loops *have* to be wrapped in brackets. Single-statement,
bracket-less bodies are not allowed.

Accounting for these differences, the surface syntax of Rust
statements and expressions is C-like. Function calls are written
`myfunc(arg1, arg2)`, operators have mostly the same name and
precedence that they have in C, comments look the same, and constructs
like `if` and `while` are available:

~~~~
# fn it_works() {}
# fn abort() {}
fn main() {
    while true {
        /* Ensure that basic math works. */
        if 2*20 > 30 {
            // Everything is OK.
            it_works();
        } else {
            abort();
        }
        break;
    }
}
~~~~

## Expression syntax

Though it isn't apparent in all code, there is a fundamental
difference between Rust's syntax and its predecessors in this family
of languages. Many constructs that are statements in C are expressions
in Rust. This allows Rust to be more expressive. For example, you might
write a piece of code like this:

~~~~
# let item = "salad";
let price;
if item == "salad" {
    price = 3.50;
} else if item == "muffin" {
    price = 2.25;
} else {
    price = 2.00;
}
~~~~

But, in Rust, you don't have to repeat the name `price`:

~~~~
# let item = "salad";
let price = if item == "salad" { 3.50 }
            else if item == "muffin" { 2.25 }
            else { 2.00 };
~~~~

Both pieces of code are exactly equivalent—they assign a value to `price`
depending on the condition that holds. Note that the semicolons are omitted
from the second snippet. This is important; the lack of a semicolon after the
last statement in a braced block gives the whole block the value of that last
expression.

Put another way, the semicolon in Rust *ignores the value of an expression*.
Thus, if the branches of the `if` had looked like `{ 4; }`, the above example
would simply assign nil (void) to `price`. But without the semicolon, each
branch has a different value, and `price` gets the value of the branch that
was taken.

This feature also works for function bodies. This function returns a boolean:

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

Rust identifiers follow the same rules as C; they start with an alphabetic
character or an underscore, and after that may contain any sequence of
alphabetic characters, numbers, or underscores. The preferred style is to
begin function, variable, and module names with a lowercase letter, using
underscores where they help readability, while beginning types with a capital
letter.

The double-colon (`::`) is used as a module separator, so
`io::println` means 'the thing named `println` in the module
named `io`.

## Variable declaration

The `let` keyword, as we've seen, introduces a local variable. Local
variables are immutable by default: `let mut` can be used to introduce
a local variable that can be reassigned.  Global constants can be
defined with `const`:

~~~~
const REPEAT: int = 5;
fn main() {
    let hi = "Hi!";
    let mut count = 0;
    while count < REPEAT {
        io::println(hi);
        count += 1;
    }
}
~~~~

Local variables may shadow earlier declarations, making the earlier variables
inaccessible.

~~~~
let my_favorite_value: float = 57.8;
let my_favorite_value: int = my_favorite_value as int;
~~~~

## Types

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

`float`
  : The largest floating-point type efficiently supported on the target
    machine.

`f32`, `f64`
  : Floating-point types with a specific size.

`char`
  : A Unicode character (32 bits).

These can be combined in composite types, which will be described in
more detail later on (the `T`s here stand for any other type):

`[T * N]`
  : Vector (like an array in other languages) with N elements.

`[mut T * N]`
  : Mutable vector with N elements.

`(T1, T2)`
  : Tuple type. Any arity above 1 is supported.

`@T`, `~T`, `&T`
  : Pointer types.

Some types can only be manipulated by pointer, never directly. For instance,
you cannot refer to a string (`str`); instead you refer to a pointer to a
string (`@str`, `~str`, or `&str`). These *dynamically-sized* types consist
of:

`fn(arg1: T1, arg2: T2) -> T3`
  : Function types.

`str`
  : String type (in UTF-8).

`[T]`
  : Vector with unknown size (also called a slice).

`[mut T]`
  : Mutable vector with unknown size.

Types can be given names with `type` declarations:

~~~~
type MonsterSize = uint;
~~~~

This will provide a synonym, `MonsterSize`, for unsigned integers. It will not
actually create a new, incompatible type—`MonsterSize` and `uint` can be used
interchangeably, and using one where the other is expected is not a type
error. Read about [single-variant enums](#single_variant_enum)
further on if you need to create a type name that's not just a
synonym.

## Using types

The `-> bool` in the `is_four` example is the way a function's return
type is written. For functions that do not return a meaningful value,
you can optionally say `-> ()`, but usually the return annotation is simply
left off, as in the `fn main() { ... }` examples we've seen earlier.

Every argument to a function must have its type declared (for example,
`x: int`). Inside the function, type inference will be able to
automatically deduce the type of most locals (generic functions, which
we'll come back to later, will occasionally need additional
annotation). Locals can be written either with or without a type
annotation:

~~~~
// The type of this vector will be inferred based on its use.
let x = [];
# vec::map(x, fn&(&&_y:int) -> int { _y });
// Explicitly say this is a vector of zero integers.
let y: [int * 0] = [];
~~~~

## Numeric literals

Integers can be written in decimal (`144`), hexadecimal (`0x90`), and
binary (`0b10010000`) base.

If you write an integer literal without a suffix (`3`, `-500`, etc.),
the Rust compiler will try to infer its type based on type annotations
and function signatures in the surrounding program. In the absence of any type
annotations at all, Rust will assume that an unsuffixed integer literal has
type `int`. It's also possible to avoid any type ambiguity by writing integer
literals with a suffix. For example:

~~~~
let x = 50;
log(error, x); // x is an int
let y = 100u;
log(error, y); // y is an uint
~~~~

Note that, in Rust, no implicit conversion between integer types
happens. If you are adding one to a variable of type `uint`, saying
`+= 1u8` will give you a type error.

Floating point numbers are written `0.0`, `1e6`, or `2.1e-4`. Without
a suffix, the literal is assumed to be of type `float`. Suffixes `f` (32-bit)
and `l` (64-bit) can be used to create literals of a specific type.

## Other literals

The nil literal is written just like the type: `()`. The keywords
`true` and `false` produce the boolean literals.

Character literals are written between single quotes, as in `'x'`. Just as in
C, Rust understands a number of character escapes, using the backslash
character, `\n`, `\r`, and `\t` being the most common.

String literals allow the same escape sequences. They are written
between double quotes (`"hello"`). Rust strings may contain newlines.

## Operators

Rust's set of operators contains very few surprises. Arithmetic is done with
`*`, `/`, `%`, `+`, and `-` (multiply, divide, remainder, plus, minus). `-` is
also a unary prefix operator that does negation. As in C, the bit operators
`>>`, `<<`, `&`, `|`, and `^` are also supported.

Note that, if applied to an integer value, `!` flips all the bits (like `~` in
C).

The comparison operators are the traditional `==`, `!=`, `<`, `>`,
`<=`, and `>=`. Short-circuiting (lazy) boolean operators are written
`&&` (and) and `||` (or).

For type casting, Rust uses the binary `as` operator.  It takes an
expression on the left side and a type on the right side and will,
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

## Syntax extensions

*Syntax extensions* are special forms that are not built into the language,
but are instead provided by the libraries. To make it clear to the reader when
a syntax extension is being used, the names of all syntax extensions end with
`!`. The standard library defines a few syntax extensions, the most useful of
which is `fmt!`, a `sprintf`-style text formatter that is expanded at compile
time.

~~~~
io::println(fmt!("%s is %d", ~"the answer", 42));
~~~~

`fmt!` supports most of the directives that [printf][pf] supports, but
will give you a compile-time error when the types of the directives
don't match the types of the arguments.

[pf]: http://en.cppreference.com/w/cpp/io/c/fprintf

You can define your own syntax extensions with the macro system, which is out
of scope of this tutorial.

# Control structures

## Conditionals

We've seen `if` pass by a few times already. To recap, braces are
compulsory, an optional `else` clause can be appended, and multiple
`if`/`else` constructs can be chained together:

~~~~
if false {
    io::println(~"that's odd");
} else if true {
    io::println(~"right");
} else {
    io::println(~"neither true nor false");
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
    else { return 0 }
}
~~~~

## Pattern matching

Rust's `match` construct is a generalized, cleaned-up version of C's
`switch` construct. You provide it with a value and a number of *arms*,
each labelled with a pattern, and the code will attempt to match each pattern
in order. For the first one that matches, the arm is executed.

~~~~
# let my_number = 1;
match my_number {
  0     => io::println("zero"),
  1 | 2 => io::println("one or two"),
  3..10 => io::println("three to ten"),
  _     => io::println("something else")
}
~~~~

There is no 'falling through' between arms, as in C—only one arm is
executed, and it doesn't have to explicitly `break` out of the
construct when it is finished.

The part to the left of the arrow `=>` is called the *pattern*. Literals are
valid patterns and will match only their own value. The pipe operator
(`|`) can be used to assign multiple patterns to a single arm. Ranges
of numeric literal patterns can be expressed with two dots, as in `M..N`. The
underscore (`_`) is a wildcard pattern that matches everything.

The patterns in an match arm are followed by a fat arrow, `=>`, then an
expression to evaluate. Each case is separated by commas. It's often
convenient to use a block expression for a case, in which case the
commas are optional.

~~~
# let my_number = 1;
match my_number {
  0 => {
    io::println("zero")
  }
  _ => {
    io::println("something else")
  }
}
~~~

`match` constructs must be *exhaustive*: they must have an arm covering every
possible case. For example, if the arm with the wildcard pattern was left off
in the above example, the typechecker would reject it.

A powerful application of pattern matching is *destructuring*, where
you use the matching to get at the contents of data types. Remember
that `(float, float)` is a tuple of two floats:

~~~~
use float::consts::pi;
fn angle(vector: (float, float)) -> float {
    match vector {
      (0f, y) if y < 0f => 1.5 * pi,
      (0f, y) => 0.5 * pi,
      (x, y) => float::atan(y / x)
    }
}
~~~~

A variable name in a pattern matches everything, *and* binds that name
to the value of the matched thing inside of the arm block. Thus, `(0f,
y)` matches any tuple whose first element is zero, and binds `y` to
the second element. `(x, y)` matches any tuple, and binds both
elements to a variable.

Any `match` arm can have a guard clause (written `if EXPR`), which is
an expression of type `bool` that determines, after the pattern is
found to match, whether the arm is taken or not. The variables bound
by the pattern are available in this guard expression.

## Let

You've already seen simple `let` bindings. `let` is also a little fancier: it
is possible to use destructuring patterns in it. For example, you can say this
to extract the fields from a tuple:

~~~~
# fn get_tuple_of_two_ints() -> (int, int) { (1, 1) }
let (a, b) = get_tuple_of_two_ints();
~~~~

This will introduce two new variables, `a` and `b`, bound to the
content of the tuple.

You may only use *irrefutable* patterns—patterns that can never fail to
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

# Functions

Like all other static declarations, such as `type`, functions can be
declared both at the top level and inside other functions (or modules,
which we'll come back to [later](#modules-and-crates)).

We've already seen several function definitions. They are introduced
with the `fn` keyword, the type of arguments are specified following
colons and the return type follows the arrow.

~~~~
fn repeat(string: &str, count: int) -> ~str {
    let mut result = ~"";
    for count.times {
        result += string;
    }
    return result;
}
~~~~

The `return` keyword immediately returns from the body of a function. It
is optionally followed by an expression to return. A function can
also return a value by having its top level block produce an
expression.

~~~~
# const copernicus: int = 0;
fn int_to_str(i: int) -> ~str {
    if i == copernicus {
        return ~"tube sock";
    } else {
        return ~"violin";
    }
}
~~~~

~~~~
# const copernicus: int = 0;
fn int_to_str(i: int) -> ~str {
    if i == copernicus { ~"tube sock" }
    else { ~"violin" }
}
~~~~

Functions that do not return a value are said to return nil, `()`,
and both the return type and the return value may be omitted from
the definition. The following two functions are equivalent.

~~~~
fn do_nothing_the_hard_way() -> () { return (); }

fn do_nothing_the_easy_way() { }
~~~~

# Basic datatypes

The core datatypes of Rust are structs, enums (tagged unions, algebraic data
types), and tuples. They are immutable by default.

~~~~
struct Point { x: float, y: float }

enum Shape {
    Circle(Point, float),
    Rectangle(Point, Point)
}
~~~~

## Structs

Rust struct types must be declared before they are used using the `struct`
syntax: `struct Name { field1: T1, field2: T2 [, ...] }`, where `T1`, `T2`,
... denote types. To construct a struct, use the same syntax, but leave off
the `struct`; for example: `Point { x: 1.0, y: 2.0 }`.

Structs are quite similar to C structs and are even laid out the same way in
memory (so you can read from a Rust struct in C, and vice-versa). The dot
operator is used to access struct fields (`mypoint.x`).

Fields that you want to mutate must be explicitly marked `mut`.

~~~~
struct Stack {
    content: ~[int],
    mut head: uint
}
~~~~

With a value of such a type, you can do `mystack.head += 1`. If `mut` were
omitted from the type, such an assignment would result in a type error.

## Struct patterns

Structs can be destructured in `match` patterns. The basic syntax is
`Name {fieldname: pattern, ...}`:
~~~~
# struct Point { x: float, y: float }
# let mypoint = Point { x: 0.0, y: 0.0 };
match mypoint {
    Point { x: 0.0, y: y } => { io::println(y.to_str());                    }
    Point { x: x, y: y }   => { io::println(x.to_str() + " " + y.to_str()); }
}
~~~~

In general, the field names of a struct do not have to appear in the same
order they appear in the type. When you are not interested in all
the fields of a struct, a struct pattern may end with `, _` (as in
`Name {field1, _}`) to indicate that you're ignoring all other fields.

## Enums

Enums are datatypes that have several alternate representations. For
example, consider the type shown earlier:

~~~~
# struct Point { x: float, y: float }
enum Shape {
    Circle(Point, float),
    Rectangle(Point, Point)
}
~~~~

A value of this type is either a Circle, in which case it contains a
point struct and a float, or a Rectangle, in which case it contains
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
    match sh {
        circle(_, size) => float::consts::pi * size * size,
        rectangle({x, y}, {x: x2, y: y2}) => (x2 - x) * (y2 - y)
    }
}
~~~~

Another example, matching nullary enum variants:

~~~~
# type point = {x: float, y: float};
# enum direction { north, east, south, west }
fn point_from_direction(dir: direction) -> point {
    match dir {
        north => {x:  0f, y:  1f},
        east  => {x:  1f, y:  0f},
        south => {x:  0f, y: -1f},
        west  => {x: -1f, y:  0f}
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
match mytup {
  (a, b, c) => log(info, a + b + (c as int))
}
~~~~

# The Rust memory model

At this junction let's take a detour to explain the concepts involved
in Rust's memory model. Rust has a very particular approach to
memory management that plays a significant role in shaping the "feel"
of the language. Understanding the memory landscape will illuminate
several of Rust's unique features as we encounter them.

Rust has three competing goals that inform its view of memory:

* Memory safety: memory that is managed by and is accessible to the
  Rust language must be guaranteed to be valid; under normal
  circumstances it must be impossible for Rust to trigger a
  segmentation fault or leak memory
* Performance: high-performance low-level code must be able to employ
  a number of allocation strategies; low-performance high-level code
  must be able to employ a single, garbage-collection-based, heap
  allocation strategy
* Concurrency: Rust must maintain memory safety guarantees, even for
  code running in parallel

## How performance considerations influence the memory model

Most languages that offer strong memory safety guarantees rely upon a
garbage-collected heap to manage all of the objects. This approach is
straightforward both in concept and in implementation, but has
significant costs. Languages that take this approach tend to
aggressively pursue ways to ameliorate allocation costs (think the
Java Virtual Machine). Rust supports this strategy with _shared
boxes_: memory allocated on the heap that may be referred to (shared)
by multiple variables.

By comparison, languages like C++ offer very precise control over
where objects are allocated. In particular, it is common to put them
directly on the stack, avoiding expensive heap allocation. In Rust
this is possible as well, and the compiler will use a clever _pointer
lifetime analysis_ to ensure that no variable can refer to stack
objects after they are destroyed.

## How concurrency considerations influence the memory model

Memory safety in a concurrent environment involves avoiding race
conditions between two threads of execution accessing the same
memory. Even high-level languages often require programmers to
correctly employ locking to ensure that a program is free of races.

Rust starts from the position that memory cannot be shared between
tasks. Experience in other languages has proven that isolating each
task's heap from the others is a reliable strategy and one that is
easy for programmers to reason about. Heap isolation has the
additional benefit that garbage collection must only be done
per-heap. Rust never "stops the world" to garbage-collect memory.

Complete isolation of heaps between tasks implies that any data
transferred between tasks must be copied. While this is a fine and
useful way to implement communication between tasks, it is also very
inefficient for large data structures.  Because of this, Rust also
employs a global _exchange heap_. Objects allocated in the exchange
heap have _ownership semantics_, meaning that there is only a single
variable that refers to them. For this reason, they are referred to as
_unique boxes_. All tasks may allocate objects on the exchange heap,
then transfer ownership of those objects to other tasks, avoiding
expensive copies.

## What to be aware of

Rust has three "realms" in which objects can be allocated: the stack,
the local heap, and the exchange heap. These realms have corresponding
pointer types: the borrowed pointer (`&T`), the shared box (`@T`),
and the unique box (`~T`). These three sigils will appear
repeatedly as we explore the language. Learning the appropriate role
of each is key to using Rust effectively.

# Boxes and pointers

In contrast to a lot of modern languages, aggregate types like records
and enums are _not_ represented as pointers to allocated memory in
Rust. They are, as in C and C++, represented directly. This means that
if you `let x = {x: 1f, y: 1f};`, you are creating a record on the
stack. If you then copy it into a data structure, the whole record is
copied, not just a pointer.

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
# fn work_with_foo_by_pointer(f: &~str) { }
let foo = ~"foo";
work_with_foo_by_pointer(&foo);
~~~~

The following shows an example of what is _not_ possible with borrowed
pointers. If you were able to write this then the pointer to `foo`
would outlive `foo` itself.

~~~~ {.ignore}
let foo_ptr;
{
    let foo = ~"foo";
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
> Unique vectors are the currently-recommended vector type for general
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
match crayons[0] {
	   bittersweet => draw_scene(crayons[0]),
       _ => ()
}
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
# fn crayon_to_str(c: crayon) -> ~str { ~"" }

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
variables declared outside the function - they do not "close over
their environment". For example, you couldn't write the following:

~~~~ {.ignore}
let foo = 10;

fn bar() -> int {
   return foo; // `bar` cannot refer to `foo`
}
~~~~

Rust also supports _closures_, functions that can access variables in
the enclosing scope.

~~~~
# import println = io::println;
fn call_closure_with_ten(b: fn(int)) { b(10); }

let captured_var = 20;
let closure = |arg| println(fmt!("captured_var=%d, arg=%d", captured_var, arg));

call_closure_with_ten(closure);
~~~~

Closures begin with the argument list between bars and are followed by
a single expression. The types of the arguments are generally omitted,
as is the return type, because the compiler can almost always infer
them. In the rare case where the compiler needs assistance though, the
arguments and return types may be annotated.

~~~~
# type mygoodness = fn(~str) -> ~str; type what_the = int;
let bloop = |well, oh: mygoodness| -> what_the { fail oh(well) };
~~~~

There are several forms of closure, each with its own role. The most
common, called a _stack closure_, has type `fn&` and can directly
access local variables in the enclosing scope.

~~~~
let mut max = 0;
(~[1, 2, 3]).map(|x| if x > max { max = x });
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

fn mk_appender(suffix: ~str) -> fn@(~str) -> ~str {
    return fn@(s: ~str) -> ~str { s + suffix };
}

fn main() {
    let shout = mk_appender(~"!");
    io::println(shout(~"hey ho, let's go"));
}
~~~~

This example uses the long closure syntax, `fn@(s: ~str) ...`,
making the fact that we are declaring a box closure explicit. In
practice boxed closures are usually defined with the short closure
syntax introduced earlier, in which case the compiler will infer
the type of closure. Thus our boxed closure example could also
be written:

~~~~
fn mk_appender(suffix: ~str) -> fn@(~str) -> ~str {
    return |s| s + suffix;
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
call_twice(|| { ~"I am an inferred stack closure"; } );
call_twice(fn&() { ~"I am also a stack closure"; } );
call_twice(fn@() { ~"I am a boxed closure"; });
call_twice(fn~() { ~"I am a unique closure"; });
fn bare_function() { ~"I am a plain function"; }
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
    debug!("%i", n);
    do_some_work(n);
});
~~~~

This is such a useful pattern that Rust has a special form of function
call that can be written more like a built-in control structure:

~~~~
# fn each(v: ~[int], op: fn(int)) {}
# fn do_some_work(i: int) { }
do each(~[1, 2, 3]) |n| {
    debug!("%i", n);
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
    debug!("I'm a task, whatever");
}
~~~~

That's nice, but look at all those bars and parentheses - that's two empty
argument lists back to back. Wouldn't it be great if they weren't
there?

~~~~
# import task::spawn;
do spawn {
   debug!("Kablam!");
}
~~~~

Empty argument lists can be omitted from `do` expressions.

## For loops

Most iteration in Rust is done with `for` loops. Like `do`,
`for` is a nice syntax for doing control flow with closures.
Additionally, within a `for` loop, `break`, `again`, and `return`
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
        println(~"found odd number!");
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
        println(~"found odd number!");
        break;
    }
}
~~~~

As an added bonus, you can use the `return` keyword, which is not
normally allowed in closures, in a block that appears as the body of a
`for` loop — this will cause a return to happen from the outer
function, not just the loop body.

~~~~
# import each = vec::each;
fn contains(v: ~[int], elt: int) -> bool {
    for each(v) |x| {
        if (x == elt) { return true; }
    }
    false
}
~~~~

`for` syntax only works with stack closures.

# Generics

## Generic functions

Throughout this tutorial, we've been defining functions that act only on
single data types. It's a burden to define such functions again and again for
every type they apply to. Thus, Rust allows functions and datatypes to have
type parameters.

~~~~
fn map<T, U>(vector: &[T], function: fn(T) -> U) -> ~[U] {
    let mut accumulator = ~[];
    for vector.each |element| {
        vec::push(accumulator, function(element));
    }
    return accumulator;
}
~~~~

When defined with type parameters, this function can be applied to any
type of vector, as long as the type of `function`'s argument and the
type of the vector's content agree with each other.

Inside a generic function, the names of the type parameters
(capitalized by convention) stand for opaque types. You can't look
inside them, but you can pass them around.

## Generic datatypes

Generic `type`, `struct`, and `enum` declarations follow the same pattern:

~~~~
struct Stack<T> {
    elements: ~[mut T]
}

enum Maybe<T> {
    Just(T),
    Nothing
}
~~~~

These declarations produce valid types like `Stack<u8>` and `Maybe<int>`.

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
> [traits](#traits) when used as type bounds, and can be
> conveniently thought of as built-in traits. In the future type
> kinds will actually be traits that the compiler has special
> knowledge about.

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
    fn chicken() -> ~str { ~"cluck cluck" }
    fn cow() -> ~str { ~"mooo" }
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
fn world() -> ~str { ~"world" }
~~~~

~~~~ {.ignore}
// main.rs
use std;
use mylib;
fn main() { io::println(~"hello " + mylib::world()); }
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
    println(~"that was easy");
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
    fn buffalo<buffalo: copy>(buffalo: buffalo) -> buffalo { buffalo }
}
fn main() {
    let buffalo: buffalo::buffalo = 1;
    buffalo::buffalo::<buffalo::buffalo>(buffalo::buffalo(buffalo));
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
type t = ~str;
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

# Traits

Traits are Rust's take on value polymorphism—the thing that
object-oriented languages tend to solve with methods and inheritance.
For example, writing a function that can operate on multiple types of
collections.

> ***Note:*** This feature is very new, and will need a few extensions to be
> applicable to more advanced use cases.

## Declaration

A trait consists of a set of methods. A method is a function that
can be applied to a `self` value and a number of arguments, using the
dot notation: `self.foo(arg1, arg2)`.

For example, we could declare the trait `to_str` for things that
can be converted to a string, with a single method of the same name:

~~~~
trait to_str {
    fn to_str() -> ~str;
}
~~~~

## Implementation

To actually implement a trait for a given type, the `impl` form
is used. This defines implementations of `to_str` for the `int` and
`~str` types.

~~~~
# trait to_str { fn to_str() -> ~str; }
impl int: to_str {
    fn to_str() -> ~str { int::to_str(self, 10u) }
}
impl ~str: to_str {
    fn to_str() -> ~str { self }
}
~~~~

Given these, we may call `1.to_str()` to get `~"1"`, or
`(~"foo").to_str()` to get `~"foo"` again. This is basically a form of
static overloading—when the Rust compiler sees the `to_str` method
call, it looks for an implementation that matches the type with a
method that matches the name, and simply calls that.

## Bounded type parameters

The useful thing about value polymorphism is that it does not have to
be static. If object-oriented languages only let you call a method on
an object when they knew exactly which sub-type it had, that would not
get you very far. To be able to call methods on types that aren't
known at compile time, it is possible to specify 'bounds' for type
parameters.

~~~~
# trait to_str { fn to_str() -> ~str; }
fn comma_sep<T: to_str>(elts: ~[T]) -> ~str {
    let mut result = ~"", first = true;
    for elts.each |elt| {
        if first { first = false; }
        else { result += ~", "; }
        result += elt.to_str();
    }
    return result;
}
~~~~

The syntax for this is similar to the syntax for specifying that a
parameter type has to be copyable (which is, in principle, another
kind of bound). By declaring `T` as conforming to the `to_str`
trait, it becomes possible to call methods from that trait on
values of that type inside the function. It will also cause a
compile-time error when anyone tries to call `comma_sep` on an array
whose element type does not have a `to_str` implementation in scope.

## Polymorphic traits

Traits may contain type parameters. This defines a trait for
generalized sequence types:

~~~~
trait seq<T> {
    fn len() -> uint;
    fn iter(fn(T));
}
impl<T> ~[T]: seq<T> {
    fn len() -> uint { vec::len(self) }
    fn iter(b: fn(T)) {
        for self.each |elt| { b(elt); }
    }
}
~~~~

Note that the implementation has to explicitly declare the type
parameter that it binds, `T`, before using it to specify its trait type. This is
needed because it could also, for example, specify an implementation
of `seq<int>`—the `of` clause *refers* to a type, rather than defining
one.

The type parameters bound by a trait are in scope in each of the
method declarations. So, re-declaring the type parameter
`T` as an explicit type parameter for `len` -- in either the trait or
the impl -- would be a compile-time error.

## The `self` type in traits

In a trait, `self` is a special type that you can think of as a
type parameter. An implementation of the trait for any given type
`T` replaces the `self` type parameter with `T`. The following
trait describes types that support an equality operation:

~~~~
trait eq {
  fn equals(&&other: self) -> bool;
}

impl int: eq {
  fn equals(&&other: int) -> bool { other == self }
}
~~~~

Notice that `equals` takes an `int` argument, rather than a `self` argument, in
an implementation for type `int`.

## Casting to a trait type

The above allows us to define functions that polymorphically act on
values of *an* unknown type that conforms to a given trait.
However, consider this function:

~~~~
# type circle = int; type rectangle = int;
# trait drawable { fn draw(); }
# impl int: drawable { fn draw() {} }
# fn new_circle() -> int { 1 }
fn draw_all<T: drawable>(shapes: ~[T]) {
    for shapes.each |shape| { shape.draw(); }
}
# let c: circle = new_circle();
# draw_all(~[c]);
~~~~

You can call that on an array of circles, or an array of squares
(assuming those have suitable `drawable` traits defined), but not
on an array containing both circles and squares.

When this is needed, a trait name can be used as a type, causing
the function to be written simply like this:

~~~~
# trait drawable { fn draw(); }
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
to a trait type:

~~~~
# type circle = int; type rectangle = int;
# trait drawable { fn draw(); }
# impl int: drawable { fn draw() {} }
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

## Trait-less implementations

If you only intend to use an implementation for static overloading,
and there is no trait available that it conforms to, you are free
to leave off the `of` clause.  However, this is only possible when you
are defining an implementation in the same module as the receiver
type, and the receiver type is a named type (i.e., an enum or a
class); [single-variant enums](#single_variant_enum) are a common
choice.

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
import libc::c_uint;

extern mod crypto {
    fn SHA1(src: *u8, sz: c_uint, out: *u8) -> *u8;
}

fn as_hex(data: ~[u8]) -> ~str {
    let mut acc = ~"";
    for data.each |byte| { acc += fmt!("%02x", byte as uint); }
    return acc;
}

fn sha1(data: ~str) -> ~str unsafe {
    let bytes = str::to_bytes(data);
    let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                            vec::len(bytes) as c_uint, ptr::null());
    return as_hex(vec::unsafe::from_buf(hash, 20u));
}

fn main(args: ~[~str]) {
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
fn SHA1(src: *u8, sz: libc::c_uint, out: *u8) -> *u8;
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
# fn as_hex(data: ~[u8]) -> ~str { ~"hi" }
fn sha1(data: ~str) -> ~str {
    unsafe {
        let bytes = str::to_bytes(data);
        let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                                vec::len(bytes), ptr::null());
        return as_hex(vec::unsafe::from_buf(hash, 20u));
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
unsafe fn kaboom() { ~"I'm harmless!"; }
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
# fn as_hex(data: ~[u8]) -> ~str { ~"hi" }
# fn x(data: ~str) -> ~str {
# unsafe {
let bytes = str::to_bytes(data);
let hash = crypto::SHA1(vec::unsafe::to_ptr(bytes),
                        vec::len(bytes), ptr::null());
return as_hex(vec::unsafe::from_buf(hash, 20u));
# }
# }
~~~~

The `str::to_bytes` function is perfectly safe: it converts a string to
a `[u8]`. This byte array is then fed to `vec::unsafe::to_ptr`, which
returns an unsafe pointer to its contents.

This pointer will become invalid as soon as the vector it points into
is cleaned up, so you should be very careful how you use it. In this
case, the local variable `bytes` outlives the pointer, so we're good.

Passing a null pointer as the third argument to `SHA1` makes it use a
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

This program uses the POSIX function `gettimeofday` to get a
microsecond-resolution timer.

~~~~
use std;
import libc::c_ulonglong;

type timeval = {mut tv_sec: c_ulonglong,
                mut tv_usec: c_ulonglong};
#[nolink]
extern mod lib_c {
    fn gettimeofday(tv: *timeval, tz: *()) -> i32;
}
fn unix_time_in_microseconds() -> u64 unsafe {
    let x = {mut tv_sec: 0 as c_ulonglong, mut tv_usec: 0 as c_ulonglong};
    lib_c::gettimeofday(ptr::addr_of(x), ptr::null());
    return (x.tv_sec as u64) * 1000_000_u64 + (x.tv_usec as u64);
}

# fn main() { assert fmt!("%?", unix_time_in_microseconds()) != ~""; }
~~~~

The `#[nolink]` attribute indicates that there's no foreign library to
link in. The standard C library is already linked with Rust programs.

A `timeval`, in C, is a struct with two 32-bit integers. Thus, we
define a record type with the same contents, and declare
`gettimeofday` to take a pointer to such a record.

The second argument to `gettimeofday` (the time zone) is not used by
this program, so it simply declares it to be a pointer to the nil
type. Since all null pointers have the same representation regardless of
their referent type, this is safe.

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
    println(~"This executes in the child task.");
    println(fmt!("%d", some_value));
}
~~~~

The argument to `task::spawn()` is a [unique
closure](#unique-closures) of type `fn~()`, meaning that it takes no
arguments and generates no return value. The effect of `task::spawn()`
is to fire up a child task that will execute the closure in parallel
with the creator.

## Communication

Now that we have spawned a child task, it would be nice if we could
communicate with it. This is done using *pipes*. Pipes are simply a
pair of endpoints, with one for sending messages and another for
receiving messages. The easiest way to create a pipe is to use
`pipes::stream`.  Imagine we wish to perform two expensive
computations in parallel.  We might write something like:

~~~~
import task::spawn;
import pipes::{stream, Port, Chan};

let (chan, port) = stream();

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
stream for sending and receiving integers:

~~~~ {.ignore}
# import pipes::stream;
let (chan, port) = stream();
~~~~

This port is where we will receive the message from the child task
once it is complete.  The channel will be used by the child to send a
message to the port.  The next statement actually spawns the child:

~~~~
# import task::{spawn};
# import comm::{Port, Chan};
# fn some_expensive_computation() -> int { 42 }
# let port = Port();
# let chan = port.chan();
do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}
~~~~

This child will perform the expensive computation send the result
over the channel.  (Under the hood, `chan` was captured by the
closure that forms the body of the child task.  This capture is
allowed because channels are sendable.)

Finally, the parent continues by performing
some other expensive computation and then waiting for the child's result
to arrive on the port:

~~~~
# import pipes::{stream, Port, Chan};
# fn some_other_expensive_computation() {}
# let (chan, port) = stream::<int>();
# chan.send(0);
some_other_expensive_computation();
let result = port.recv();
~~~~

## Creating a task with a bi-directional communication path

A very common thing to do is to spawn a child task where the parent
and child both need to exchange messages with each other. The
function `std::comm::DuplexStream()` supports this pattern.  We'll
look briefly at how it is used.

To see how `spawn_conversation()` works, we will create a child task
that receives `uint` messages, converts them to a string, and sends
the string in response.  The child terminates when `0` is received.
Here is the function that implements the child task:

~~~~
# import std::comm::DuplexStream;
# import pipes::{Port, Chan};
fn stringifier(channel: DuplexStream<~str, uint>) {
    let mut value: uint;
    loop {
        value = channel.recv();
        channel.send(uint::to_str(value, 10u));
        if value == 0u { break; }
    }
}
~~~~

The implementation of `DuplexStream` supports both sending and
receiving. The `stringifier` function takes a `DuplexStream` that can
send strings (the first type parameter) and receive `uint` messages
(the second type parameter). The body itself simply loops, reading
from the channel and then sending its response back.  The actual
response itself is simply the strified version of the received value,
`uint::to_str(value)`.

Here is the code for the parent task:

~~~~
# import std::comm::DuplexStream;
# import pipes::{Port, Chan};
# import task::spawn;
# fn stringifier(channel: DuplexStream<~str, uint>) {
#     let mut value: uint;
#     loop {
#         value = channel.recv();
#         channel.send(uint::to_str(value, 10u));
#         if value == 0u { break; }
#     }
# }
# fn main() {

let (from_child, to_child) = DuplexStream();

do spawn || {
    stringifier(to_child);
};

from_child.send(22u);
assert from_child.recv() == ~"22";

from_child.send(23u);
from_child.send(0u);

assert from_child.recv() == ~"23";
assert from_child.recv() == ~"0";

# }
~~~~

The parent task first calls `DuplexStream` to create a pair of bidirectional endpoints. It then uses `task::spawn` to create the child task, which captures one end of the communication channel.  As a result, both parent
and child can send and receive data to and from the other.

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
