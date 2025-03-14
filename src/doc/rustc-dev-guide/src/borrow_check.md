# MIR borrow check

The borrow check is Rust's "secret sauce" – it is tasked with
enforcing a number of properties:

- That all variables are initialized before they are used.
- That you can't move the same value twice.
- That you can't move a value while it is borrowed.
- That you can't access a place while it is mutably borrowed (except through
  the reference).
- That you can't mutate a place while it is immutably borrowed.
- etc

The borrow checker operates on the MIR. An older implementation operated on the
HIR. Doing borrow checking on MIR has several advantages:

- The MIR is *far* less complex than the HIR; the radical desugaring
  helps prevent bugs in the borrow checker. (If you're curious, you
  can see
  [a list of bugs that the MIR-based borrow checker fixes here][47366].)
- Even more importantly, using the MIR enables ["non-lexical lifetimes"][nll],
  which are regions derived from the control-flow graph.

[47366]: https://github.com/rust-lang/rust/issues/47366
[nll]: https://rust-lang.github.io/rfcs/2094-nll.html

### Major phases of the borrow checker

The borrow checker source is found in
[the `rustc_borrowck` crate][b_c]. The main entry point is
the [`mir_borrowck`] query.

[b_c]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/index.html
[`mir_borrowck`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/fn.mir_borrowck.html

- We first create a **local copy** of the MIR. In the coming steps,
  we will modify this copy in place to modify the types and things to
  include references to the new regions that we are computing.
- We then invoke [`replace_regions_in_mir`] to modify our local MIR.
  Among other things, this function will replace all of the [regions](./appendix/glossary.md#region)
  in the MIR with fresh [inference variables](./appendix/glossary.md#inf-var).
- Next, we perform a number of
  [dataflow analyses](./appendix/background.md#dataflow) that
  compute what data is moved and when.
- We then do a [second type check](borrow_check/type_check.md) across the MIR:
  the purpose of this type check is to determine all of the constraints between
  different regions.
- Next, we do [region inference](borrow_check/region_inference.md), which computes
  the values of each region — basically, the points in the control-flow graph where
  each lifetime must be valid according to the constraints we collected.
- At this point, we can compute the "borrows in scope" at each point.
- Finally, we do a second walk over the MIR, looking at the actions it
  does and reporting errors. For example, if we see a statement like
  `*a + 1`, then we would check that the variable `a` is initialized
  and that it is not mutably borrowed, as either of those would
  require an error to be reported. Doing this check requires the results of all
  the previous analyses.

[`replace_regions_in_mir`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/nll/fn.replace_regions_in_mir.html
