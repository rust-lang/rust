# Syntax Basics

FIXME: briefly mention syntax extentions, #fmt

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

    fn main() {
        if 1 < 2 {
            while false { call_a_function(10 * 4); }
        } else if 4 < 3 || 3 < 4 {
            // Comments are C++-style too
        } else {
            /* Multi-line comment syntax */
        }
    }

## Expression syntax

Though it isn't apparent in most everyday code, there is a fundamental
difference between Rust's syntax and the predecessors in this family
of languages. A lot of thing that are statements in C are expressions
in Rust. This allows for useless things like this (which passes
nil—the void type—to a function):

    a_function(while false {});

But also useful things like this:

    let x = if the_stars_align() { 4 }
            else if something_else() { 3 }
            else { 0 };

This piece of code will bind the variable `x` to a value depending on
the conditions. Note the condition bodies, which look like `{
expression }`. The lack of a semicolon after the last statement in a
braced block gives the whole block the value of that last expression.
If the branches of the `if` had looked like `{ 4; }`, the above
example would simply assign nil (void) to `x`. But without the
semicolon, each branch has a different value, and `x` gets the value
of the branch that was taken.

This also works for function bodies. This function returns a boolean:

    fn is_four(x: int) -> bool { x == 4 }

In short, everything that's not a declaration (`let` for variables,
`fn` for functions, etcetera) is an expression.

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

Rust will normally emit warning about unused variables. These can be
suppressed by using a variable name that starts with an underscore.

    fn this_warns(x: int) {}
    fn this_doesnt(_x: int) {}

## Variable declaration

The `let` keyword, as we've seen, introduces a local variable. Global
constants can be defined with `const`:

    import std;
    const repeat: uint = 5u;
    fn main() {
        let count = 0u;
        while count < repeat {
            std::io::println("Hi!");
            count += 1u;
        }
    }

## Types

The `-> bool` in the last example is the way a function's return type
is written. For functions that do not return a meaningful value (these
conceptually return nil in Rust), you can optionally say `-> ()` (`()`
is how nil is written), but usually the return annotation is simply
left off, as in the `fn main() { ... }` examples we've seen earlier.

Every argument to a function must have its type declared (for example,
`x: int`). Inside the function, type inference will be able to
automatically deduce the type of most locals (generic functions, which
we'll come back to later, will occasionally need additional
annotation). Locals can be written either with or without a type
annotation:

    // The type of this vector will be inferred based on its use.
    let x = [];
    // Explicitly say this is a vector of integers.
    let y: [int] = [];

The basic types are written like this:

`()`
: Nil, the type that has only a single value.  

`bool`
: Boolean type..  

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
: String type. A string contains a utf-8 encoded sequence of characters.

These can be combined in composite types, which will be described in
more detail later on (the `T`s here stand for any other type):

`[T]`
: Vector type.  

`[mutable T]`
: Mutable vector type.  

`(T1, T2)`
: Tuple type. Any arity above 1 is supported.  

`{fname1: T1, fname2: T2}`
: Record type.  

`fn(arg1: T1, arg2: T2) -> T3`
: Function type.  

`@T`, `~T`, `*T`
: Pointer types.  

`obj { fn method1() }`
: Object type.

Types can be given names with `type` declarations:

    type monster_size = uint;

This will provide a synonym, `monster_size`, for unsigned integers. It
will not actually create a new type—`monster_size` and `uint` can be
used interchangeably, and using one where the other is expected is not
a type error. Read about [single-variant tags][svt] further on if you
need to create a type name that's not just a synonym.

[svt]: data.html#single_variant_tag

## Literals

Integers can be written in decimal (`144`), hexadecimal (`0x90`), and
binary (`0b10010000`) base. Without suffix, an integer literal is
considered to be of type `int`. Add a `u` (`144u`) to make it a `uint`
instead. Literals of the fixed-size integer types can be created by
the literal with the type name (`255u8`, `50i64`, etc).

Note that, in Rust, no implicit conversion between integer types
happens. If you are adding one to a variable of type `uint`, you must
type `v += 1u`—saying `+= 1` will give you a type error.

Floating point numbers are written `0.0`, `1e6`, or `2.1e-4`. Without
suffix, the literal is assumed to be of type `float`. Suffixes `f32`
and `f64` can be used to create literals of a specific type. The
suffix `f` can be used to write `float` literals without a dot or
exponent: `3f`.

The nil literal is written just like the type: `()`. The keywords
`true` and `false` produce the boolean literals.

Character literals are written between single quotes, as in `'x'`. You
may put non-ascii characters between single quotes (your source file
should be encoded as utf-8 in that case). Rust understands a number of
character escapes, using the backslash character:

`\n`
: A newline (unicode character 32).

`\r`
: A carriage return (13).

`\t`
: A tab character (9).

`\\`, `\'`, `\"`
: Simply escapes the following character.

`\xHH`, `\uHHHH`, `\UHHHHHHHH`
: Unicode escapes, where the `H` characters are the hexadecimal digits that form the character code.

String literals allow the same escape sequences. They are written
between double quotes (`"hello"`). Rust strings may contain newlines.
When a newline is preceded by a backslash, it, and all white space
following it, will not appear in the resulting string literal.

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

    let message = badness < 10 ? "error" : "FATAL ERROR";

For type casting, Rust uses the binary `as` operator, which has a
precedence between the bitwise combination operators (`&`, `|`, `^`)
and the comparison operators. It takes an expression on the left side,
and a type on the right side, and will, if a meaningful conversion
exists, convert the result of the expression to the given type.

    let x: float = 4.0;
    let y: uint = x as uint;
    assert y == 4u;

## Attributes

Every definition can be annotated with attributes. Attributes are meta
information that can serve a variety of purposes. One of those is
conditional compilation:

    #[cfg(target_os = "win32")]
    fn register_win_service() { /* ... */ }

This will cause the function to vanish without a trace during
compilation on a non-Windows platform. Attributes always look like
`#[attr]`, where `attr` can be simply a name (as in `#[test]`, which
is used by the [built-in test framework](test.html)), a name followed
by `=` and then a literal (as in `#[license = "BSD"]`, which is a
valid way to annotate a Rust program as being released under a
BSD-style license), or a name followed by a comma-separated list of
nested attributes, as in the `cfg` example above.

An attribute without a semicolon following it applies to the
definition that follows it. When terminated with a semicolon, it
applies to the current context. The above example could also be
written like this:

    fn register_win_service() {
        #[cfg(target_os = "win32")];
        /* ... */
    }
