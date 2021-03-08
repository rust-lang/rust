# The THIR

<!-- toc -->

The THIR ("Typed High-Level Intermediate Representation"), previously HAIR for
"High-Level Abstract IR", is another IR used by rustc that is generated after
[type checking]. It is (as of <!-- date: 2021-03 --> March 2021) only used for
[MIR construction] and [exhaustiveness checking], but
[it may also soon be used for unsafety checking][thir-unsafeck] as a replacement
for the current MIR unsafety checker.

[type checking]: ./type-checking.md
[MIR construction]: ./mir/construction.md
[exhaustiveness checking]: ./pat-exhaustive-checking.md
[thir-unsafeck]: https://github.com/rust-lang/compiler-team/issues/402

As the name might suggest, the THIR is a lowered version of the [HIR] where all
the types have been filled in, which is possible after type checking has completed.
But it has some other interesting features that distinguish it from HIR:
- like the MIR, the THIR only represents bodies, i.e. "executable code"; this includes
  function bodies, but also `const` initializers, for example. Consequently, the THIR
  has no representation for items like `struct`s or `trait`s.
- a body of THIR is only stored temporarily and is dropped as soon as it's no longer
  needed, as opposed to being stored until the end of the compilation process (which
  is what is done with the HIR).
- besides making the types of all nodes available, the THIR also has additional
  desugaring compared to the HIR. For example, automatic references and dereferences
  are made explicit, and method calls and overloaded operators are converted into
  plain function calls. Destruction scopes are also made explicit.

[HIR]: ./hir.md

The THIR lives in [`rustc_mir_build::thir`][thir]. To construct a `thir::Expr`,
you can use the `build_thir` function, passing in the memory arena where the THIR
will be allocated. Dropping this arena will result in the THIR being destroyed:
this is useful to keep peak memory in check, as having a THIR representation of
all bodies of a crate in memory at the same time would be very heavy.

[thir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/index.html
