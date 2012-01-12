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

    fn dead_end() -> ! { fail; }

This helps the compiler avoid spurious error messages. For example,
the following code would be a type error if `dead_end` would be
expected to return.

    # fn can_go_left() -> bool { true }
    # fn can_go_right() -> bool { true }
    # enum dir { left; right; }
    # fn dead_end() -> ! { fail; }
    let dir = if can_go_left() { left }
              else if can_go_right() { right }
              else { dead_end(); };

## Closures

Named functions, like those in the previous section, do not close over
their environment. Rust also includes support for closures, which are
functions that can access variables in the scope in which they are
created.

There are several forms of closures, each with its own role. The most
common type is called a 'block', this is a closure which has full
access to its environment.

    fn call_block_with_ten(b: block(int)) { b(10); }
    
    let x = 20;    
    call_block_with_ten({|arg|
        #info("x=%d, arg=%d", x, arg);
    });

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

    use std;
    
    fn mk_appender(suffix: str) -> fn@(str) -> str {
        let f = fn@(s: str) -> str { s + suffix };
        ret f;
    }
    
    fn main() {
        let shout = mk_appender("!");
        std::io::println(shout("hey ho, let's go"));
    }

### Closure compatibility

A nice property of Rust closures is that you can pass any kind of
closure (as long as the arguments and return types match) to functions
that expect a `block`. Thus, when writing a higher-order function that
wants to do nothing with its function argument beyond calling it, you
should almost always specify the type of that argument as `block`, so
that callers have the flexibility to pass whatever they want.

    fn call_twice(f: block()) { f(); f(); }
    call_twice({|| "I am a block"; });
    call_twice(fn@() { "I am a boxed closure"; });
    fn bare_function() { "I am a plain function"; }
    call_twice(bare_function);

### Unique closures

<a name="unique"></a>

Unique closures, written `fn~` in analogy to the `~` pointer type (see
next section), hold on to things that can safely be sent between
processes. They copy the values they close over, much like boxed
closures, but they also 'own' them—meaning no other code can access
them. Unique closures mostly exist to for spawning new
[tasks](task.html).

### Shorthand syntax

The compact syntax used for blocks (`{|arg1, arg2| body}`) can also
be used to express boxed and unique closures in situations where the
closure style can be unambiguously derived from the context. Most
notably, when calling a higher-order function you do not have to use
the long-hand syntax for the function you're passing, since the
compiler can look at the argument type to find out what the parameter
types are.

As a further simplification, if the final parameter to a function is a
closure, the closure need not be placed within parenthesis. You could,
for example, write...

    let doubled = vec::map([1, 2, 3]) {|x| x*2};

`vec::map` is a function in the core library that applies its last
argument to every element of a vector, producing a new vector.

Even when a closure takes no parameters, you must still write the bars
for the parameter list, as in `{|| ...}`.

## Binding

Partial application is done using the `bind` keyword in Rust.

    let daynum = bind vec::position(_, ["mo", "tu", "we", "do",
                                        "fr", "sa", "su"]);

Binding a function produces a boxed closure (`fn@` type) in which some
of the arguments to the bound function have already been provided.
`daynum` will be a function taking a single string argument, and
returning the day of the week that string corresponds to (if any).

## Iteration

Functions taking blocks provide a good way to define non-trivial
iteration constructs. For example, this one iterates over a vector
of integers backwards:

    fn for_rev(v: [int], act: block(int)) {
        let i = vec::len(v);
        while (i > 0u) {
            i -= 1u;
            act(v[i]);
        }
    }

To run such an iteration, you could do this:

    # fn for_rev(v: [int], act: block(int)) {}
    for_rev([1, 2, 3], {|n| log(error, n); });

Making use of the shorthand where a final closure argument can be
moved outside of the parentheses permits the following, which
looks quite like a normal loop:

    # fn for_rev(v: [int], act: block(int)) {}
    for_rev([1, 2, 3]) {|n|
        log(error, n);
    }

Note that, because `for_rev()` returns unit type, no semicolon is
needed when the final closure is pulled outside of the parentheses.
