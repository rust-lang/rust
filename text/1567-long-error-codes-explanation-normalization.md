
Start Date: 2016-01-04

- RFC PR: [rust-lang/rfcs#1567](https://github.com/rust-lang/rfcs/pull/1567)
- Rust Issue: N/A

# Summary

Rust has extend error messages that explain each error in more detail. We've been writing lots of them, which is good, but they're written in different styles, which is bad. This RFC intends to fix this inconsistency by providing a template for these long-form explanations to follow.

# Motivation

Long error codes explanations are a very important part of Rust. Having an explanation of what failed helps to understand the error and is appreciated by Rust developers of all skill levels. Providing an unified template is needed in order to help people who would want to write ones as well as people who read them.

# Detailed design

Here is what I propose:

## Error description

Provide a more detailed error message. For example:

```rust
extern crate a;
extern crate b as a;
```

We get the `E0259` error code which says "an extern crate named `a` has already been imported in this module" and the error explanation says: "The name chosen for an external crate conflicts with another external crate that has been imported into the current module.".

## Minimal example

Provide an erroneous code example which directly follows `Error description`. The erroneous example will be helpful for the `How to fix the problem`. Making it as simple as possible is really important in order to help readers to understand what the error is about. A comment should be added with the error on the same line where the errors occur. Example:

```rust
type X = u32<i32>; // error: type parameters are not allowed on this type
```

If the error comments is too long to fit 80 columns, split it up like this, so the next line start at the same column of the previous line:

```rust
type X = u32<'static>; // error: lifetime parameters are not allowed on
                       //        this type
```

And if the sample code is too long to write an effective comment, place your comment on the line before the sample code:

```rust
// error: lifetime parameters are not allowed on this type
fn super_long_function_name_and_thats_problematic() {}
```

Of course, it the comment is too long, the split rules still applies.

## Error explanation

Provide a full explanation about "__why__ you get the error" and some leads on __how__ to fix it. If needed, use additional code snippets to improve your explanations.

## How to fix the problem

This part will show how to fix the error that we saw previously in the `Minimal example`, with comments explaining how it was fixed.

## Additional information

Some details which might be useful for the users, let's take back `E0109` example. At the end, the supplementary explanation is the following: "Note that type parameters for enum-variant constructors go after the variant, not after the enum (`Option::None::<u32>`, not `Option::<u32>::None`).". It provides more information, not directly linked to the error, but it might help user to avoid doing another error.

## Template

In summary, the template looks like this:

```rust
E000: r##"
[Error description]

Example of erroneous code:

\```compile_fail
[Minimal example]
\```

[Error explanation]

\```
[How to fix the problem]
\```

[Optional Additional information]
```

Now let's take a full example:

> E0409: r##"
> An "or" pattern was used where the variable bindings are not consistently bound
> across patterns.
>
> Example of erroneous code:
>
> ```compile_fail
> let x = (0, 2);
> match x {
>     (0, ref y) | (y, 0) => { /* use y */} // error: variable `y` is bound with
>                                           //        different mode in pattern #2
>                                           //        than in pattern #1
>     _ => ()
> }
> ```
>
> Here, `y` is bound by-value in one case and by-reference in the other.
>
> To fix this error, just use the same mode in both cases.
> Generally using `ref` or `ref mut` where not already used will fix this:
>
> ```ignore
> let x = (0, 2);
> match x {
>     (0, ref y) | (ref y, 0) => { /* use y */}
>     _ => ()
> }
> ```
>
> Alternatively, split the pattern:
>
> ```
> let x = (0, 2);
> match x {
>     (y, 0) => { /* use y */ }
>     (0, ref y) => { /* use y */}
>     _ => ()
> }
> ```
> "##,

# Drawbacks

This will make contributing slighty more complex, as there are rules to follow, whereas right now there are none.

# Alternatives

Not having error codes explanations following a common template.

# Unresolved questions

None.
