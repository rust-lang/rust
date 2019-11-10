# How to write documentation

This chapter covers not only how to write documentation but specifically
how to write **good** documentation.  Something to keep in mind when
writing documentation is that your audience is not just yourself but others
who simply don't have the context you do.  It is important to be as clear
as you can, and as complete as possible.  As a rule of thumb: the more
documentation you write for your crate the better.  If an item is public
then it should be documented.

## Basic structure

It is recommended that each item's documentation follows this basic structure:

```text
[short sentence explaining what it is]

[more detailed explanation]

[at least one code example that users can copy/paste to try it]

[even more advanced explanations if necessary]
```

This basic structure should be straightforward to follow when writing your
documentation and, while you might think that a code example is trivial,
the examples are really important because they can help your users to
understand what an item is, how it is used, and for what purpose it exists.

Let's see an example coming from the [standard library] by taking a look at the
[`std::env::args()`][env::args] function:

``````text
Returns the arguments which this program was started with (normally passed
via the command line).

The first element is traditionally the path of the executable, but it can be
set to arbitrary text, and may not even exist. This means this property should
not be relied upon for security purposes.

On Unix systems shell usually expands unquoted arguments with glob patterns
(such as `*` and `?`). On Windows this is not done, and such arguments are
passed as-is.

# Panics

The returned iterator will panic during iteration if any argument to the
process is not valid unicode. If this is not desired,
use the [`args_os`] function instead.

# Examples

```
use std::env;

// Prints each argument on a separate line
for argument in env::args() {
    println!("{}", argument);
}
```

[`args_os`]: ./fn.args_os.html
``````

As you can see, it follows the structure detailed above: it starts with a short
sentence explaining what the functions does, then it provides more information
and finally provides a code example.

## Markdown

`rustdoc` is using the [commonmark markdown specification]. You might be
interested into taking a look at their website to see what's possible to do.

## Lints

To be sure that you didn't miss any item without documentation or code examples,
you can take a look at the rustdoc lints [here][rustdoc-lints].

[standard library]: https://doc.rust-lang.org/stable/std/index.html
[env::args]: https://doc.rust-lang.org/stable/std/env/fn.args.html
[commonmark markdown specification]: https://commonmark.org/
[rustdoc-lints]: lints.md
