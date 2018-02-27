- Feature Name: `macro-at-most-once-rep`
- Start Date: 2018-01-17
- RFC PR: [rust-lang/rfcs#2298](https://github.com/rust-lang/rfcs/pull/2298)
- Rust Issue: [rust-lang/rust#48075](https://github.com/rust-lang/rust/issues/48075)


Summary
-------

Add a repetition specifier to macros to repeat a pattern at most once: `$(pat)?`. Here, `?` behaves like `+` or `*` but represents at most one repetition of `pat`.

Motivation
----------

There are two specific use cases in mind.

## Macro rules with optional parts

Currently, you just have to write two rules and possibly have one "desugar" to the other.

```rust
macro_rules! foo {
  (do $b:block) => {
    $b
  }
  (do $b1:block and $b2:block) => {
    foo!($b1)
    $b2
  }
}
```

Under this RFC, one would simply write:

```rust
macro_rules! foo {
  (do $b1:block $(and $b2:block)?) => {
    $b1
    $($b2)?
  }
}
```

## Trailing commas

Currently, the best way to make a rule tolerate trailing commas is to create another identical rule that has a comma at the end:

```rust
macro_rules! foo {
  ($(pat),+,) => { foo!( $(pat),+ ) };
  ($(pat),+) => {
    // do stuff
  }
}
```

or to allow multiple trailing commas:

```rust
macro_rules! foo {
  ($(pat),+ $(,)*) => {
    // do stuff
  }
}
```

This is unergonomic and clutters up macro definitions needlessly. Under this RFC, one would simply write:

```rust
macro_rules! foo {
  ($(pat),+ $(,)?) => {
    // do stuff
  }
}
```

Guide-level explanation
-----------------------

In Rust macros, you specify some "rules" which define how the macro is used and what it transforms to. For each rule, there is a pattern and a body:

```rust
macro_rules! foo {
  (pattern) => { body }
}
```

The pattern portion is composed of zero or more subpatterns concatenated together. One possible subpattern is to repeat another subpattern some number of times. This is useful when writing variadic macros (e.g. `println`):

```rust
macro_rules! println {
  // Takes a variable number of arguments after the template
  ($tempate:expr, $($args:expr),*) => { ... }
}
```
which can be invoked like so:
```rust
println!("")           // 0 args
println!("", foo)      // 1 args
println!("", foo, bar) // 2 args
...
```

The `*` in the pattern of this example indicates "0 or more repetitions". One can also use `+` for "at _least_ one repetition" or `?` for "at _most_ one repetition".

In the body of a rule, one can specify to repeat some code for every occurence of the pattern in the invokation:

```rust
macro_rules! foo {
  ($($pat:expr),*) => {
    $(
      println!("{}", $pat)
    )* // Repeat for each `expr` passed to the macro
  }
}
```

The same can be done for `+` and `?`.

The `?` operator is particularly useful for making macro rules with optional components in the invocation or for making macros tolerate trailing commas.

Reference-level explanation
---------------------------

`?` is identical to `+` and `*` in use except that it represents "at most once" repetition.

Introducing `?` into the grammar for macro repetition introduces an easily fixable ambiguity, as noted by @kennytm [here](https://internals.rust-lang.org/t/pre-rfc-at-most-one-repetition-macro-patterns/6557/2?u=mark-i-m):

  > There is ambiguity: $($x:ident)?+ today matches a?b?c and not a+. Fortunately this is easy to resolve: you just look one more token ahead and always treat ?* and ?+ to mean separate by the question mark token.

Drawbacks
---------
While there are grammar ambiguities, they can be easily fixed.

Also, for patterns that use `*`, `?` is not a perfect solution: `$(pat),* $(,)?` still allows `,` which is a bit weird. However, this is still an improvement over `$(pat),* $(,)*` which allows `,,,,,`.

Rationale and Alternatives
--------------------------

The implementation of `?` ought to be very similar to `+` and `*`. Only the parser needs to change; to the author's knowledge, it would not be technically difficult to implement, nor would it add much complexity to the compiler.

The `?` character is chosen because
- As noted above, there are grammar ambiguities, but they can be easily fixed
- It is consistent with common regex syntax, as are `+` and `*`
- It intuitively expresses "this pattern is optional"

One alternative to alleviate the trailing comma paper cut is to allow trailing commas automatically for any pattern repetitions. This would be a breaking change. Also, it would allow trailing commas in potentially unwanted places. For example:

```rust
macro_rules! foo {
  ($($pat:expr),*; $(foo),*) => {
    $(
      println!("{}", $pat)
    )* // Repeat for each `expr` passed to the macro
  }
}
```
would allow
```rust
foo! {
  x,; foo
}
```

Also, rather than have `?` be a repetition operator, we could have the compiler do a "copy/paste" of the rule and insert the optional pattern. Implementation-wise, this might reuse less code than the proposal. Also, it's probably less easy to teach; this RFC is very easy to teach because `?` is another operator like `+` or `*`.

We could use another symbol other than `?`, but it's not clear what other options might be better. `?` has the advantage of already being known in common regex syntax as "optional".

It has also been suggested to add `{M, N}` (at least `M` but no more than `N`) either in addition to or as an alternative to `?`. Like `?`, `{M, N}` is common regex syntax and has the same implementation difficulty level. However, it's not clear how useful such a pattern would be. In particular, we can't think of any other language to include this sort of "partially-variadic" argument list. It is also questionable why one would want to _syntactically_ repeat some piece of code between `M` and `N` times. Thus, this RFC does not propose to add `{M, N}` at this time (though we note that it is forward-compatible).

Finally, we could do nothing and wait for macros 2.0. However, it will be a while (possibly years) before that lands in stable rust. The current implementation and proposals are not very well-defined yet. Having something until that time would be nice to fix this paper cut. This proposal does not add a lot of complexity, but does nicely fill the gap.

Unresolved Questions
--------------------

- Should the `?` Kleene operator accept a separator? Adding a separator is completely meaningless (since we don't accept trailing separators, and `?` can accept "at most one" repetition), but allowing it is consistent with `+` and `*`. Currently, we allow a separator. We could also make it an error or lint.
