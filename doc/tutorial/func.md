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
In Rust, these are annotated with return type `!`:

    fn dead_end() -> ! { fail; }

This helps the compiler avoid spurious error messages. For example,
the following code would be a type error if `dead_end` would be
expected to return.

    # fn can_go_left() -> bool { true }
    # fn can_go_right() -> bool { true }
    # tag dir { left; right; }
    # fn dead_end() -> ! { fail; }
    let dir = if can_go_left() { left }
              else if can_go_right() { right }
              else { dead_end(); };

## Closures

Named rust functions, like those in the previous section, do not close
over their environment. Rust also includes support for closures, which
are anonymous functions that can access the variables that were in
scope at the time the closure was created.  Closures are represented
as the pair of a function pointer (as in C) and the environment, which
is where the values of the closed over variables are stored.  Rust
includes support for three varieties of closure, each with different
costs and capabilities:

- Stack closures (written `block`) store their environment in the
  stack frame of their creator; they are very lightweight but cannot
  be stored in a data structure.
- Boxed closures (written `fn@`) store their environment in a
  [shared box](data#shared-box).  These are good for storing within
  data structures but cannot be sent to another task.
- Unique closures (written `fn~`) store their environment in a
  [unique box](data#unique-box).  These are limited in the kinds of
  data that they can close over so that they can be safely sent
  between tasks.  As with any unique pointer, copying a unique closure
  results in a deep clone of the environment.
  
Both boxed closures and unique closures are subtypes of stack
closures, meaning that wherever a stack closure type appears, a boxed
or unique closure value can be used.  This is due to the restrictions
placed on the use of stack closures, which ensure that all operations
on a stack closure are also safe on any kind of closure.

### Working with closures

Closures are specified by writing an inline, anonymous function
declaration.  For example, the following code creates a boxed closure:

    let plus_two = fn@(x: int) -> int {
        ret x + 2;
    };
    
Creating a unique closure is very similar:

    let plus_two_uniq = fn~(x: int) -> int {
        ret x + 2;
    };
    
Stack closures can be created in a similar way; however, because stack
closures literally point into their creator's stack frame, they can
only be used in a very specific way.  Stack closures may be passed as
parameters and they may be called, but they may not be stored into
local variables or fields.  Creating a stack closure can therefore be
done using a syntax like the following:

    let doubled = vec::map([1, 2, 3], block(x: int) -> int {
        x * 2
    });
    
Here the `vec::map()` is the standard higher-order map function, which
applies the closure to each item in the vector and returns a new
vector containing the results.
    
### Shorthand syntax

The syntax in the previous section was very explicit; it fully
specifies the kind of closure as well as the type of every parameter
and the return type.  In practice, however, closures are often used as
parameters to functions, and all of these details can be inferred.
Therefore, we support a shorthand syntax similar to Ruby or Smalltalk
blocks, which looks as follows:

    let doubled = vec::map([1, 2, 3], {|x| x*2});
 
Here the vertical bars after the open brace `{` indicate that this is
a closure.  A list of parameters appears between the bars.  The bars
must always be present: if there are no arguments, then simply write
`{||...}`.

As a further simplification, if the final parameter to a function is a
closure, the closure need not be placed within parenthesis.
Therefore, one could write

    let doubled = vec::map([1, 2, 3]) {|x| x*2};
   
This form is often easier to parse as it involves less nesting.  

## Binding

Partial application is done using the `bind` keyword in Rust.

    let daynum = bind std::vec::position(_, ["mo", "tu", "we", "do",
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
        let i = std::vec::len(v);
        while (i > 0u) {
            i -= 1u;
            act(v[i]);
        }
    }

To run such an iteration, you could do this:

    # fn for_rev(v: [int], act: block(int)) {}
    for_rev([1, 2, 3], {|n| log n; });

Making use of the shorthand where a final closure argument can be
moved outside of the parentheses permits the following, which
looks quite like a normal loop:

    # fn for_rev(v: [int], act: block(int)) {}
    for_rev([1, 2, 3]) {|n|
        log n;
    }

Note that, because `for_rev()` returns unit type, no semicolon is
needed when the final closure is pulled outside of the parentheses.

## Capture clauses

When creating a boxed or unique closure, the default is to copy in the
values of any closed over variables.  But sometimes, particularly if a
value is large or expensive to copy, you would like to *move* the
value into the closure instead.  Rust supports this via the use of a
capture clause, which lets you specify precisely whether each variable
used in the closure is copied or moved.

As an example, let's assume we had some type of unique tree type:

    tag tree<T> = tree_rec<T>;
    type tree_rec<T> = ~{left: option<tree>, right: option<tree>, val: T};

Now if we have a function like the following:

    let some_tree: tree<T> = ...;
    let some_closure = fn~() {
        ... use some_tree in some way ...
    };
    
Here the variable `some_tree` is used within the closure body, so a
deep copy will be performed.  This can become quite expensive if the
tree is large.  If we know that `some_tree` will not be used again,
we could avoid this expense by making use of a capture clause like so:

    let some_tree: tree<T> = ...;
    let some_closure = fn~[move some_tree]() {
        ... use some_tree in some way ...
    };

This is particularly useful when moving data into [child tasks](task).
