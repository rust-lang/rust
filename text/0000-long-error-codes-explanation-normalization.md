
Start Date: 2016-01-04

RFC PR:

Rust Issue: N/A

# Summary

Long error codes explanations haven't been normalized yet. This RFC intends to do it in order to uniformize them.

# Motivation

Long error codes explanations are a very important part of Rust. Having an explanation of what failed helps to understand the error and is appreciated by Rust developers of all skill levels. Providing an unified template is needed in order to help people who would want to write ones as well as people who read them.

# Detailed design

Here is the template I propose:

## First point

Giving a little more detailed error message. For example, the `E0109` says "type parameters are not allowed on this type" and the error explanation says: "You tried to give a type parameter to a type which doesn't need it.".

## Second point

Giving an erroneous code example which directly follows `First point`. It'll be helpful for the `Forth point`. Making it as simple as possible is really important in order to help readers to understand what the error is about. A comment should be added with the error on the same line that the errors happen. Example:

 ```Rust
 type X = u32<i32>; // error: type parameters are not allowed on this type
 ```
 
 If the error comments is too long to fit 80 columns, split it up like this, so the next line start at the same column of the previous line:
 
 ```Rust
 type X = u32<'static>; // error: lifetime parameters are not allowed on
                        //        this type
 ```
 
 And if the code line is just too long to make a correct comment, put your comment before it:
 
```Rust
// error: lifetime parameters are not allowed on this type
fn super_long_function_name_and_thats_problematic() {}
```
 
Of course, it the comment is too long, the split rules still applies.

## Third point

Providing a full explanation about "__why__ you get the error" and some leads on __how__ to fix it. If needed, add little code examples to improve your explanations.

## Fourth point

This part will show how to fix the error that we saw previously in the `Second point`, with comments explaining how it was fixed.

## Fifth point

Some details which might be useful for the users, let's take back `E0109` example. At the end, the supplementary explanation is the following: "Note that type parameters for enum-variant constructors go after the variant, not after the enum (`Option::None::<u32>`, not `Option::<u32>::None`).". It provides more information, not directly linked to the error, but it might help user to avoid doing another error.

## Template

So in final, it should like this:

```Rust
E000: r##"
[First point] Example of erroneous code:

\```compile_fail
[Second point]
\```

[Third point]

\```
[Fourth point]
\```

[Optional Fifth point]
```

Now let's take a full example:

> E0264: r##"
> An unknown external lang item was used. Example of erroneous code:
>
> ```compile_fail
> #![feature(lang_items)]
> extern "C" {
>     #[lang = "cake"] // error: unknown external lang item: `cake`
>     fn cake();
> }
> ```
>
> A list of available external lang items is available in
> `src/librustc/middle/weak_lang_items.rs`. Example:
>
> ```
> #![feature(lang_items)]
> extern "C" {
>     #[lang = "panic_fmt"] // ok!
>     fn cake();
> }
> ```
> "##,

# Drawbacks

None.

# Alternatives

Not having error codes explanations uniformized.

# Unresolved questions

None.
