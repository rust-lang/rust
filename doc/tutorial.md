% Rust Language Tutorial

# Introduction

Rust is a programming language with a focus on type safety, memory
safety, concurrency and performance. It is intended for writing
large-scale, high-performance software while preventing several
classes of common errors. Rust has a sophisticated memory model that
encourages efficient data structures and safe concurrency patterns,
forbidding invalid memory accesses that would otherwise cause
segmentation faults. It is statically typed and compiled ahead of
time.

As a multi-paradigm language, Rust supports writing code in
procedural, functional and object-oriented styles. Some of its
pleasant high-level features include:

* **Pattern matching and algebraic data types (enums).** As
  popularized by functional languages, pattern matching on ADTs
  provides a compact and expressive way to encode program logic.
* **Type inference.** Type annotations on local variable
  declarations are optional.
* **Task-based concurrency.** Rust uses lightweight tasks that do
  not share memory.
* **Higher-order functions.** Rust's efficient and flexible closures
  are heavily relied on to provide iteration and other control
  structures
* **Parametric polymorphism (generics).** Functions and types can be
  parameterized over type variables with optional trait-based type
  constraints.
* **Trait polymorphism.** Rust's type system features a unique
  combination of type classes and object-oriented interfaces.

## Scope

This is an introductory tutorial for the Rust programming language. It
covers the fundamentals of the language, including the syntax, the
type system and memory model, and generics.  [Additional
tutorials](#what-next) cover specific language features in greater
depth.

It assumes the reader is familiar with the basic concepts of
programming, and has programmed in one or more other languages
before. It will often make comparisons to other languages,
particularly those in the C family.

## Conventions

Throughout the tutorial, words that indicate language keywords or
identifiers defined in example code are displayed in `code font`.

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
$ wget http://dl.rust-lang.org/dist/rust-0.4.tar.gz
$ tar -xzf rust-0.4.tar.gz
$ cd rust-0.4
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
[tarball]: http://dl.rust-lang.org/dist/rust-0.4.tar.gz

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
  : Pointer types. See [Boxes and pointers](#boxes-and-pointers) for an explanation of what `@`, `~`, and `&` mean.

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
# vec::map(x, fn&(_y: &int) -> int { *_y });
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
enum Direction {
    North,
    East,
    South,
    West
}
~~~~

This will define `North`, `East`, `South`, and `West` as constants,
all of which have type `Direction`.

When an enum is C-like, that is, when none of the variants have
parameters, it is possible to explicitly set the discriminator values
to an integer value:

~~~~
enum Color {
  Red = 0xff0000,
  Green = 0x00ff00,
  Blue = 0x0000ff
}
~~~~

If an explicit discriminator is not specified for a variant, the value
defaults to the value of the previous variant plus one. If the first
variant does not have a discriminator, it defaults to 0. For example,
the value of `North` is 0, `East` is 1, etc.

When an enum is C-like the `as` cast operator can be used to get the
discriminator's value.

<a name="single_variant_enum"></a>

There is a special case for enums with a single variant. These are
used to define new types in such a way that the new name is not just a
synonym for an existing type, but its own distinct type. If you say:

~~~~
enum GizmoId = int;
~~~~

That is a shorthand for this:

~~~~
enum GizmoId { GizmoId(int) }
~~~~

Enum types like this can have their content extracted with the
dereference (`*`) unary operator:

~~~~
# enum GizmoId = int;
let my_gizmo_id = GizmoId(10);
let id_int: int = *my_gizmo_id;
~~~~

## Enum patterns

For enum types with multiple variants, destructuring is the only way to
get at their contents. All variant constructors can be used as
patterns, as in this definition of `area`:

~~~~
# type Point = {x: float, y: float};
# enum Shape { Circle(Point, float), Rectangle(Point, Point) }
fn area(sh: Shape) -> float {
    match sh {
        Circle(_, size) => float::consts::pi * size * size,
        Rectangle({x, y}, {x: x2, y: y2}) => (x2 - x) * (y2 - y)
    }
}
~~~~

Another example, matching nullary enum variants:

~~~~
# type Point = {x: float, y: float};
# enum Direction { North, East, South, West }
fn point_from_direction(dir: Direction) -> Point {
    match dir {
        North => {x:  0f, y:  1f},
        East  => {x:  1f, y:  0f},
        South => {x:  0f, y: -1f},
        West  => {x: -1f, y:  0f}
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
enum Crayon {
    Almond, AntiqueBrass, Apricot,
    Aquamarine, Asparagus, AtomicTangerine,
    BananaMania, Beaver, Bittersweet
}

// A stack vector of crayons
let stack_crayons: &[Crayon] = &[Almond, AntiqueBrass, Apricot];
// A local heap (shared) vector of crayons
let local_crayons: @[Crayon] = @[Aquamarine, Asparagus, AtomicTangerine];
// An exchange heap (unique) vector of crayons
let exchange_crayons: ~[Crayon] = ~[BananaMania, Beaver, Bittersweet];
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
# enum Crayon { Almond, AntiqueBrass, Apricot,
#               Aquamarine, Asparagus, AtomicTangerine,
#               BananaMania, Beaver, Bittersweet };
# fn draw_scene(c: Crayon) { }

let crayons = ~[BananaMania, Beaver, Bittersweet];
match crayons[0] {
    Bittersweet => draw_scene(crayons[0]),
    _ => ()
}
~~~~

By default, vectors are immutable—you can not replace their elements.
The type written as `~[mut T]` is a vector with mutable
elements. Mutable vector literals are written `~[mut]` (empty) or `~[mut
1, 2, 3]` (with elements).

~~~~
# enum Crayon { Almond, AntiqueBrass, Apricot,
#               Aquamarine, Asparagus, AtomicTangerine,
#               BananaMania, Beaver, Bittersweet };

let crayons = ~[mut BananaMania, Beaver, Bittersweet];
crayons[0] = AtomicTangerine;
~~~~

The `+` operator means concatenation when applied to vector types.

~~~~
# enum Crayon { Almond, AntiqueBrass, Apricot,
#               Aquamarine, Asparagus, AtomicTangerine,
#               BananaMania, Beaver, Bittersweet };

let my_crayons = ~[Almond, AntiqueBrass, Apricot];
let your_crayons = ~[BananaMania, Beaver, Bittersweet];

let our_crayons = my_crayons + your_crayons;
~~~~

The `+=` operator also works as expected, provided the assignee
lives in a mutable slot.

~~~~
# enum Crayon { Almond, AntiqueBrass, Apricot,
#               Aquamarine, Asparagus, AtomicTangerine,
#               BananaMania, Beaver, Bittersweet };

let mut my_crayons = ~[Almond, AntiqueBrass, Apricot];
let your_crayons = ~[BananaMania, Beaver, Bittersweet];

my_crayons += your_crayons;
~~~~

## Vector and string methods

Both vectors and strings support a number of useful
[methods](#implementation).  While we haven't covered methods yet,
most vector functionality is provided by methods, so let's have a
brief look at a few common ones.

~~~
# use io::println;
# enum Crayon {
#     Almond, AntiqueBrass, Apricot,
#     Aquamarine, Asparagus, AtomicTangerine,
#     BananaMania, Beaver, Bittersweet
# }
# fn unwrap_crayon(c: Crayon) -> int { 0 }
# fn eat_crayon_wax(i: int) { }
# fn store_crayon_in_nasal_cavity(i: uint, c: Crayon) { }
# fn crayon_to_str(c: Crayon) -> ~str { ~"" }

let crayons = ~[Almond, AntiqueBrass, Apricot];

// Check the length of the vector
assert crayons.len() == 3;
assert !crayons.is_empty();

// Iterate over a vector, obtaining a pointer to each element
for crayons.each |crayon| {
    let delicious_crayon_wax = unwrap_crayon(*crayon);
    eat_crayon_wax(delicious_crayon_wax);
}

// Map vector elements
let crayon_names = crayons.map(|v| crayon_to_str(*v));
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
# use println = io::println;
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
(~[1, 2, 3]).map(|x| if *x > max { max = *x });
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
extern mod std;

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
integers, passing in a pointer to each integer in the vector:

~~~~
fn each(v: ~[int], op: fn(v: &int)) {
   let mut n = 0;
   while n < v.len() {
       op(&v[n]);
       n += 1;
   }
}
~~~~

The reason we pass in a *pointer* to an integer rather than the
integer itself is that this is how the actual `each()` function for
vectors works.  Using a pointer means that the function can be used
for vectors of any type, even large records that would be impractical
to copy out of the vector on each iteration.  As a caller, if we use a
closure to provide the final operator argument, we can write it in a
way that has a pleasant, block-like structure.

~~~~
# fn each(v: ~[int], op: fn(v: &int)) { }
# fn do_some_work(i: int) { }
each(~[1, 2, 3], |n| {
    debug!("%i", *n);
    do_some_work(*n);
});
~~~~

This is such a useful pattern that Rust has a special form of function
call that can be written more like a built-in control structure:

~~~~
# fn each(v: ~[int], op: fn(v: &int)) { }
# fn do_some_work(i: int) { }
do each(~[1, 2, 3]) |n| {
    debug!("%i", *n);
    do_some_work(*n);
}
~~~~

The call is prefixed with the keyword `do` and, instead of writing the
final closure inside the argument list it is moved outside of the
parenthesis where it looks visually more like a typical block of
code. The `do` expression is purely syntactic sugar for a call that
takes a final closure argument.

`do` is often used for task spawning.

~~~~
use task::spawn;

do spawn() || {
    debug!("I'm a task, whatever");
}
~~~~

That's nice, but look at all those bars and parentheses - that's two empty
argument lists back to back. Wouldn't it be great if they weren't
there?

~~~~
# use task::spawn;
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
fn each(v: ~[int], op: fn(v: &int) -> bool) {
   let mut n = 0;
   while n < v.len() {
       if !op(&v[n]) {
           break;
       }
       n += 1;
   }
}
~~~~

And using this function to iterate over a vector:

~~~~
# use each = vec::each;
# use println = io::println;
each(~[2, 4, 8, 5, 16], |n| {
    if *n % 2 != 0 {
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
# use each = vec::each;
# use println = io::println;
for each(~[2, 4, 8, 5, 16]) |n| {
    if *n % 2 != 0 {
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
# use each = vec::each;
fn contains(v: ~[int], elt: int) -> bool {
    for each(v) |x| {
        if (*x == elt) { return true; }
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
fn map<T, U>(vector: &[T], function: fn(v: &T) -> U) -> ~[U] {
    let mut accumulator = ~[];
    for vec::each(vector) |element| {
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
inside them, but you can pass them around.  Note that instances of
generic types are almost always passed by pointer.  For example, the
parameter `function()` is supplied with a pointer to a value of type
`T` and not a value of type `T` itself.  This ensures that the
function works with the broadest set of types possible, since some
types are expensive or illegal to copy and pass by value.

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
fn head<T: Copy>(v: ~[T]) -> T { v[0] }
~~~~

When instantiating a generic function, you can only instantiate it
with types that fit its kinds. So you could not apply `head` to a
resource type. Rust has several kinds that can be used as type bounds:

* `Copy` - Copyable types. All types are copyable unless they
  are classes with destructors or otherwise contain
  classes with destructors.
* `Send` - Sendable types. All types are sendable unless they
  contain shared boxes, closures, or other local-heap-allocated
  types.
* `Const` - Constant types. These are types that do not contain
  mutable fields nor shared boxes.

> ***Note:*** Rust type kinds are syntactically very similar to
> [traits](#traits) when used as type bounds, and can be
> conveniently thought of as built-in traits. In the future type
> kinds will actually be traits that the compiler has special
> knowledge about.

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
trait ToStr {
    fn to_str() -> ~str;
}
~~~~

## Implementation

To actually implement a trait for a given type, the `impl` form
is used. This defines implementations of `to_str` for the `int` and
`~str` types.

~~~~
# trait ToStr { fn to_str() -> ~str; }
impl int: ToStr {
    fn to_str() -> ~str { int::to_str(self, 10u) }
}
impl ~str: ToStr {
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
# trait ToStr { fn to_str() -> ~str; }
fn comma_sep<T: ToStr>(elts: ~[T]) -> ~str {
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

Traits may contain type parameters. A trait for
generalized sequence types is:

~~~~
trait Seq<T> {
    fn len() -> uint;
    fn iter(b: fn(v: &T));
}
impl<T> ~[T]: Seq<T> {
    fn len() -> uint { vec::len(self) }
    fn iter(b: fn(v: &T)) {
        for vec::each(self) |elt| { b(elt); }
    }
}
~~~~

The implementation has to explicitly declare the type
parameter that it binds, `T`, before using it to specify its trait type. Rust requires this declaration because the `impl` could also, for example, specify an implementation of `seq<int>`. The trait type -- appearing after the colon in the `impl` -- *refers* to a type, rather than defining one.

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
trait Eq {
  fn equals(&&other: self) -> bool;
}

impl int: Eq {
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
# type Circle = int; type Rectangle = int;
# trait Drawable { fn draw(); }
# impl int: Drawable { fn draw() {} }
# fn new_circle() -> int { 1 }
fn draw_all<T: Drawable>(shapes: ~[T]) {
    for shapes.each |shape| { shape.draw(); }
}
# let c: Circle = new_circle();
# draw_all(~[c]);
~~~~

You can call that on an array of circles, or an array of squares
(assuming those have suitable `drawable` traits defined), but not
on an array containing both circles and squares.

When this is needed, a trait name can be used as a type, causing
the function to be written simply like this:

~~~~
# trait Drawable { fn draw(); }
fn draw_all(shapes: ~[Drawable]) {
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
# type Circle = int; type Rectangle = int;
# trait Drawable { fn draw(); }
# impl int: Drawable { fn draw() {} }
# fn new_circle() -> int { 1 }
# fn new_rectangle() -> int { 2 }
# fn draw_all(shapes: ~[Drawable]) {}
let c: Circle = new_circle();
let r: Rectangle = new_rectangle();
draw_all(~[c as Drawable, r as Drawable]);
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
to leave off the type after the colon.  However, this is only possible when you
are defining an implementation in the same module as the receiver
type, and the receiver type is a named type (i.e., an enum or a
class); [single-variant enums](#single_variant_enum) are a common
choice.

# Modules and crates

The Rust namespace is divided into modules. Each source file starts
with its own module.

## Local modules

The `mod` keyword can be used to open a new, local module. In the
example below, `chicken` lives in the module `farm`, so, unless you
explicitly import it, you must refer to it by its long name,
`farm::chicken`.

~~~~
#[legacy_exports]
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

Having compiled a crate that contains the `#[crate_type = "lib"]`
attribute, you can use it in another crate with a `use`
directive. We've already seen `extern mod std` in several of the
examples, which loads in the [standard library][std].

[std]: http://doc.rust-lang.org/doc/std/index/General.html

`use` directives can appear in a crate file, or at the top level of a
single-file `.rs` crate. They will cause the compiler to search its
library search path (which you can extend with `-L` switch) for a Rust
crate library with the right name.

It is possible to provide more specific information when using an
external crate.

~~~~ {.ignore}
extern mod myfarm (name = "farm", vers = "2.7");
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
extern mod mylib;
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
extern mod std;
use io::println;
fn main() {
    println(~"that was easy");
}
~~~~

It is also possible to import just the name of a module (`use
std::list;`, then use `list::find`), to import all identifiers exported
by a given module (`use io::*`), or to import a specific set
of identifiers (`use math::{min, max, pi}`).

You can rename an identifier when importing using the `=` operator:

~~~~
use prnt = io::println;
~~~~

## Exporting

By default, a module exports everything that it defines. This can be
restricted with `export` directives at the top of the module or file.

~~~~
mod enc {
    export encrypt, decrypt;
    const SUPER_SECRET_NUMBER: int = 10;
    fn encrypt(n: int) -> int { n + SUPER_SECRET_NUMBER }
    fn decrypt(n: int) -> int { n - SUPER_SECRET_NUMBER }
}
~~~~

This defines a rock-solid encryption algorithm. Code outside of the
module can refer to the `enc::encrypt` and `enc::decrypt` identifiers
just fine, but it does not have access to `enc::super_secret_number`.

## Namespaces

Rust uses three different namespaces: one for modules, one for types,
and one for values. This means that this code is valid:

~~~~
#[legacy_exports]
mod buffalo {
    type buffalo = int;
    fn buffalo<buffalo>(+buffalo: buffalo) -> buffalo { buffalo }
}
fn main() {
    let buffalo: buffalo::buffalo = 1;
    buffalo::buffalo::<buffalo::buffalo>(buffalo::buffalo(buffalo));
}
~~~~

You don't want to write things like that, but it *is* very practical
to not have to worry about name clashes between types, values, and
modules.

## Resolution

The resolution process in Rust simply goes up the chain of contexts,
looking for the name in each context. Nested functions and modules
create new contexts inside their parent function or module. A file
that's part of a bigger crate will have that crate's context as its
parent context.

Identifiers can shadow each other. In this program, `x` is of type
`int`:

~~~~
type MyType = ~str;
fn main() {
    type MyType = int;
    let x: MyType;
}
~~~~

An `use` directive will only import into the namespaces for which
identifiers are actually found. Consider this example:

~~~~
mod foo { fn bar() {} }
fn baz() {
    let bar = 10u;

    {
        use foo::bar;
        let quux = bar;
    }
}
~~~~

When resolving the type name `bar` in the `quux` definition, the
resolver will first look at local block context for `baz`. This has an
import named `bar`, but that's function, not a value, So it continues
to the `baz` function context and finds a value named `bar` defined
there.

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

# What next?

Now that you know the essentials, check out any of the additional
tutorials on individual topics.

* [Borrowed pointers][borrow]
* [Tasks and communication][tasks]
* [Macros][macros]
* [The foreign function interface][ffi]

There is further documentation on the [wiki], including articles about
[unit testing] in Rust, [documenting][rustdoc] and [packaging][cargo]
Rust code, and a discussion of the [attributes] used to apply metada
to code.

[borrow]: tutorial-borrowed-ptr.html
[tasks]: tutorial-tasks.html
[macros]: tutorial-macros.html
[ffi]: tutorial-ffi.html

[wiki]: https://github.com/mozilla/rust/wiki/Docs
[unit testing]: https://github.com/mozilla/rust/wiki/Doc-unit-testing
[rustdoc]: https://github.com/mozilla/rust/wiki/Doc-using-rustdoc
[cargo]: https://github.com/mozilla/rust/wiki/Doc-using-cargo-to-manage-packages
[attributes]: https://github.com/mozilla/rust/wiki/Doc-attributes

[pound-rust]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust
