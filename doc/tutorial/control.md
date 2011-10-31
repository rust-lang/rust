# Control structures

## Conditionals

We've seen `if` pass by a few times already. To recap, braces are
compulsory, an optional `else` clause can be appended, and multiple
`if`/`else` constructs can be chained together:

    if false {
        std::io::println("that's odd");
    } else if true {
        std::io::println("right");
    } else {
        std::io::println("neither true nor false");
    }

The condition given to an `if` construct *must* be of type boolean (no
implicit conversion happens). If the arms return a value, this value
must be of the same type for every arm in which control reaches the
end of the block:

    fn signum(x: int) -> int {
        if x < 0 { -1 }
        else if x > 0 { 1 }
        else { ret 0; }
    }

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

    alt my_number {
      0       { std::io::println("zero"); }
      1 | 2   { std::io::println("one or two"); }
      3 to 10 { std::io::println("three to ten"); }
      _       { std::io::println("something else"); }
    }

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

    fn angle(vec: (float, float)) -> float {
        alt vec {
          (0f, y) when y < 0f { 1.5 * std::math::pi }
          (0f, y) { 0.5 * std::math::pi }
          (x, y) { std::math::atan(y / x) }
        }
    }

A variable name in a pattern matches everything, *and* binds that name
to the value of the matched thing inside of the arm block. Thus, `(0f,
y)` matches any tuple whose first element is zero, and binds `y` to
the second element. `(x, y)` matches any tuple, and binds both
elements to a variable.

Any `alt` arm can have a guard clause (written `when EXPR`), which is
an expression of type `bool` that determines, after the pattern is
found to match, whether the arm is taken or not. The variables bound
by the pattern are available in this guard expression.

## Destructuring let

To a limited extent, it is possible to use destructuring patterns when
declaring a variable with `let`. For example, you can say this to
extract the fields from a tuple:

    let (a, b) = get_tuple_of_two_ints();

This will introduce two new variables, `a` and `b`, bound to the
content of the tuple.

You may only use irrevocable patterns in let bindings, though. Things
like literals, which only match a specific value, are not allowed.

## Loops

`while` produces a loop that runs as long as its given condition
(which must have type `bool`) evaluates to true. Inside a loop, the
keyword `break` can be used to abort the loop, and `cont` can be used
to abort the current iteration and continue with the next.

    let x = 5;
    while true {
        x += x - 3;
        if x % 5 == 0 { break; }
        std::io::println(std::int::str(x));
    }

This code prints out a weird sequence of numbers and stops as soon as
it finds one that can be divided by five.

When iterating over a vector, use `for` instead.

    for elt in ["red", "green", "blue"] {
        std::io::println(elt);
    }

This will go over each element in the given vector (a three-element
vector of strings, in this case), and repeatedly execute the body with
`elt` bound to the current element. You may add an optional type
declaration (`elt: str`) for the iteration variable if you want.

For more involved iteration, such as going over the elements of a hash
table, Rust uses higher-order functions. We'll come back to those in a
moment.

## Failure

The `fail` keyword causes the current [task][tasks] to fail. You use
it to indicate unexpected failure, much like you'd use `exit(1)` in a
C program, except that in Rust, it is possible for other tasks to
handle the failure, allowing the program to continue running.

`fail` takes an optional argument, which must have type `str`. Trying
to access a vector out of bounds, or running a pattern match with no
matching clauses, both result in the equivalent of a `fail`.

[tasks]: FIXME

## Logging

Rust has a built-in logging mechanism, using the `log` statement.
Logging is polymorphic—any type of value can be logged, and the
runtime will do its best to output a textual representation of the
value.

    log "hi";
    log (1, [2.5, -1.8]);

By default, you *will not* see the output of your log statements. The
environment variable `RUST_LOG` controls which log statements actually
get output. It can contain a comma-separated list of paths for modules
that should be logged. For example, running `rustc` with
`RUST_LOG=rustc::front::attr` will turn on logging in its attribute
parser. If you compile a program `foo.rs`, you can set `RUST_LOG` to
`foo` to enable its logging.

Turned-off `log` statements impose minimal overhead on the code that
contains them, so except in code that needs to be really, really fast,
you should feel free to scatter around debug logging statements, and
leave them in.

For interactive debugging, you often want unconditional logging. For
this, use `log_err` instead of `log` [FIXME better name].
