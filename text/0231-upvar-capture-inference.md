- Start Date: 2014-09-09
- RFC PR: [rust-lang/rfcs#231](https://github.com/rust-lang/rfcs/pull/231)
- Rust Issue: [rust-lang/rust#16640](https://github.com/rust-lang/rust/issues/16640)

# Summary

The `||` unboxed closure form should be split into two forms—`||` for nonescaping closures and `move ||` for escaping closures—and the capture clauses and self type specifiers should be removed.

# Motivation

Having to specify `ref` and the capture mode for each unboxed closure is inconvenient (see Rust PR rust-lang/rust#16610). It would be more convenient for the programmer if the type of the closure and the modes of the upvars could be inferred. This also eliminates the "line-noise" syntaxes like `|&:|`, which are arguably unsightly.

Not all knobs can be removed, however—the programmer must manually specify whether each closure is escaping or nonescaping. To see this, observe that no sensible default for the closure `|| (*x).clone()` exists: if the function is nonescaping, it's a closure that returns a copy of `x` every time but does not move `x` into it; if the function is escaping, it's a closure that returns a copy of `x` and takes ownership of `x`.

Therefore, we need two forms: one for *nonescaping* closures and one for *escaping* closures. Nonescaping closures are the commonest, so they get the `||` syntax that we have today, and a new `move ||` syntax will be introduced for escaping closures.

# Detailed design

For unboxed closures specified with `||`, the capture modes of the free variables are determined as follows:

1. Any variable which is closed over and borrowed mutably is by-reference and mutably borrowed.

2. Any variable of a type that does not implement `Copy` which is moved within the closure is captured by value.

3. Any other variable which is closed over is by-reference and immutably borrowed.

The trait that the unboxed closure implements is `FnOnce` if any variables were moved *out* of the closure; otherwise `FnMut` if there are any variables that are closed over and mutably borrowed; otherwise `Fn`.

The `ref` prefix for unboxed closures is removed, since it is now essentially implied.

We introduce a new grammar production, `move ||`. The value returned by a `move ||` implements `FnOnce`, `FnMut`, or `Fn`, as determined above; thus, for example, `move |x: int, y| x + y` produces an unboxed closure that implements the `Fn(int, int) -> int` trait (and thus the `FnOnce(int, int) -> int` trait by inheritance). Free variables referenced by a `move ||` closure are always captured by value.

In the trait reference grammar, we will change the `|&:|` sugar to `Fn()`, the `|&mut:|` sugar to `FnMut()`, and the `|:|` sugar to `FnOnce()`. Thus what was before written `fn foo<F:|&mut: int, int| -> int>()` will be `fn foo<F:FnMut(int, int) -> int>()`.

It is important to note that the trait reference syntax and closure construction syntax are purposefully distinct. This is because either the `||` form or the `move ||` form can construct any of `FnOnce`, `FnMut`, or `Fn` closures.

# Drawbacks

1. Having two syntaxes for closures could be seen as unfortunate.

2. `move` becomes a keyword.

# Alternatives

1. Keep the status quo: `|:|`/`|&mut:`/`|&:|` are the only ways to create unboxed closures, and `ref` must be used to get by-reference upvars.

2. Use some syntax other than `move ||` for escaping closures.

3. Keep the  `|:|`/`|&mut:`/`|&:|` syntax only for trait reference sugar.

4. Use `fn()` syntax for trait reference sugar.

# Unresolved questions

There may be unforeseen complications in doing the inference.
