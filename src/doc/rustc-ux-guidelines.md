% Rustc UX guidelines

Don't forget the user. Whether human or another program, such as an IDE, a
good user experience with the compiler goes a long way into making developer
lives better. We don't want users to be baffled by compiler output or
learn arcane patterns to compile their program.

## Error, Warning, Help, Note Messages

When the compiler detects a problem, it can emit either an error, warning,
note, or help message.

An `error` is emitted when the compiler detects a problem that makes it unable
 to compile the program, either because the program is invalid or the
 programmer has decided to make a specific `warning` into an error.

A `warning` is emitted when the compiler detects something odd about a
program. For instance, dead code and unused `Result` values.

A `help` is emitted following either an `error` or `warning` giving extra
information to the user about how to solve their problem.

A `note` is for identifying additional circumstances and parts of the code
that lead to a warning or error. For example, the borrow checker will note any
previous conflicting borrows.

* Write in plain simple English. If your message, when shown on a – possibly
small – screen (which hasn't been cleaned for a while), cannot be understood
by a normal programmer, who just came out of bed after a night partying, it's
too complex.
* `Errors` and `Warnings` should not suggest how to fix the problem. A `Help`
message should be emitted instead.
* `Error`, `Warning`, `Note`, and `Help` messages start with a lowercase
letter and do not end with punctuation.
* Error messages should be succinct. Users will see these error messages many
times, and more verbose descriptions can be viewed with the `--explain` flag.
That said, don't make it so terse that it's hard to understand.
* The word "illegal" is illegal. Prefer "invalid" or a more specific word
instead.
* Errors should document the span of code where they occur – the `span_..`
methods allow to easily do this. Also `note` other spans that have contributed
to the error if the span isn't too large.
* When emitting a message with span, try to reduce the span to the smallest
amount possible that still signifies the issue
* Try not to emit multiple error messages for the same error. This may require
detecting duplicates.
* When the compiler has too little information for a specific error message,
lobby for annotations for library code that allow adding more. For example see
`#[on_unimplemented]`. Use these annotations when available!
* Keep in mind that Rust's learning curve is rather steep, and that the
compiler messages are an important learning tool.

## Error Explanations

Error explanations are long form descriptions of error messages provided with
the compiler. They are accessible via the `--explain` flag. Each explanation
comes with an example of how to trigger it and advice on how to fix it.

Long error codes explanations are a very important part of Rust. Having an
explanation of what failed helps to understand the error and is appreciated by
Rust developers of all skill levels.

* All of them are accessible [online](http://doc.rust-lang.org/error-index.html),
  which are auto-generated from rustc source code in different places:
  [librustc](https://github.com/rust-lang/rust/blob/master/src/librustc/diagnostics.rs),
  [librustc_borrowck](https://github.com/rust-lang/rust/blob/master/src/librustc_borrowck/diagnostics.rs),
  [librustc_const_eval](https://github.com/rust-lang/rust/blob/master/src/librustc_const_eval/diagnostics.rs),
  [librustc_lint](https://github.com/rust-lang/rust/blob/master/src/librustc_lint/types.rs),
  [librustc_metadata](https://github.com/rust-lang/rust/blob/master/src/librustc_metadata/diagnostics.rs),
  [librustc_mir](https://github.com/rust-lang/rust/blob/master/src/librustc_mir/diagnostics.rs),
  [librustc_passes](https://github.com/rust-lang/rust/blob/master/src/librustc_passes/diagnostics.rs),
  [librustc_privacy](https://github.com/rust-lang/rust/blob/master/src/librustc_privacy/diagnostics.rs),
  [librustc_resolve](https://github.com/rust-lang/rust/blob/master/src/librustc_resolve/diagnostics.rs),
  [librustc_trans](https://github.com/rust-lang/rust/blob/master/src/librustc_trans/diagnostics.rs),
  [librustc_typeck](https://github.com/rust-lang/rust/blob/master/src/librustc_typeck/diagnostics.rs).
* Explanations have full markdown support. Use it, especially to highlight
code with backticks.
* When talking about the compiler, call it `the compiler`, not `Rust` or
`rustc`.

Note: The following sections are mostly a repaste of [RFC 1567](https://github.com/rust-lang/rfcs/blob/master/text/1567-long-error-codes-explanation-normalization.md).

### Template

Long error descriptions should match the following template. The next few
sections of this document describe what each section is about.

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

### Error description

Provide a more detailed error message. For example:

```rust
extern crate a;
extern crate b as a;
```

We get the `E0259` error code which says "an extern crate named `a` has already been imported in this module" and the error explanation says: "The name chosen for an external crate conflicts with another external crate that has been imported into the current module.".

### Minimal example

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

### Error explanation

Provide a full explanation about "__why__ you get the error" and some leads on __how__ to fix it. If needed, use additional code snippets to improve your explanations.

### How to fix the problem

This part will show how to fix the error that we saw previously in the `Minimal example`, with comments explaining how it was fixed.

### Additional information

Some details which might be useful for the users, let's take back `E0109` example. At the end, the supplementary explanation is the following: "Note that type parameters for enum-variant constructors go after the variant, not after the enum (`Option::None::<u32>`, not `Option::<u32>::None`).". It provides more information, not directly linked to the error, but it might help user to avoid doing another error.

### Full Example

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

## Compiler Flags

* Flags should be orthogonal to each other. For example, if we'd have a
json-emitting variant of multiple actions `foo` and `bar`, an additional
--json flag is better than adding `--foo-json` and `--bar-json`.
* Always give options a long descriptive name, if only for better
understandable compiler scripts.
* The `--verbose` flag is for adding verbose information to `rustc` output
when not compiling a program. For example, using it with the `--version` flag
gives information about the hashes of the code.
* Experimental flags and options must be guarded behind the `-Z unstable-options` flag.
