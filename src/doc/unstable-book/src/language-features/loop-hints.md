# `loop_hints`

The tracking issue for this feature is: [#156874]

[#156874]: https://github.com/rust-lang/rust/issues/156874

------

<!--
FIXME(#159429): temporarily renamed `#[unroll]` to mitigate a nameres ambiguity
-->

Loop unrolling can be a powerful optimization but like inlining, it is sometimes useful to
manually provide hints to optimizations.

`#[rustc_unroll]` will encourage unrolling of a loop.

`#[rustc_unroll(full)]` is a stronger hint and can cause optimizations to completely ignore the code
side growth from repeating a loop body.

`#[rustc_unroll(never)]` is a strong hint to not unroll the loop at all. Note that other loop
optimizations may still be applied.

`#[rustc_unroll(N)]` is a hint to unroll `N` iterations of the loop.

In all cases these are just hints and may be ignored. But unlike function inlining hints,
loops tend to be heavily modified during compilation, which can make obeying hints challenging.
If the attribute doesn't do what you want, please file an issue.
