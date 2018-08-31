# MIR borrow check

The borrow check is Rust's "secret sauce" – it is tasked with
enforcing a number of properties:

- That all variables are initialized before they are used.
- That you can't move the same value twice.
- That you can't move a value while it is borrowed.
- That you can't access a place while it is mutably borrowed (except through
  the reference).
- That you can't mutate a place while it is shared borrowed.
- etc

At the time of this writing, the code is in a state of transition. The
"main" borrow checker still works by processing [the HIR](hir.html),
but that is being phased out in favor of the MIR-based borrow checker.
Doing borrow checking on MIR has two key advantages:

- The MIR is *far* less complex than the HIR; the radical desugaring
  helps prevent bugs in the borrow checker. (If you're curious, you
  can see
  [a list of bugs that the MIR-based borrow checker fixes here][47366].)
- Even more importantly, using the MIR enables ["non-lexical lifetimes"][nll],
  which are regions derived from the control-flow graph.

[47366]: https://github.com/rust-lang/rust/issues/47366
[nll]: http://rust-lang.github.io/rfcs/2094-nll.html

### Major phases of the borrow checker

The borrow checker source is found in
[the `rustc_mir::borrow_check` module][b_c]. The main entry point is
the `mir_borrowck` query. At the time of this writing, MIR borrowck can operate
in several modes, but this text will describe only the mode when NLL is enabled
(what you get with `#![feature(nll)]`).

[b_c]: https://github.com/rust-lang/rust/tree/master/src/librustc_mir/borrow_check

The overall flow of the borrow checker is as follows:

- We first create a **local copy** C of the MIR. In the coming steps,
  we will modify this copy in place to modify the types and things to
  include references to the new regions that we are computing.
- We then invoke `nll::replace_regions_in_mir` to modify this copy C.
  Among other things, this function will replace all of the regions in
  the MIR with fresh [inference variables](./appendix/glossary.html).
  - (More details can be found in [the regionck section](./mir/regionck.html).)
- Next, we perform a number of [dataflow
  analyses](./appendix/background.html#dataflow)
  that compute what data is moved and when. The results of these analyses
  are needed to do both borrow checking and region inference.
- Using the move data, we can then compute the values of all the regions in the
  MIR.
  - (More details can be found in [the NLL section](./mir/regionck.html).)
- Finally, the borrow checker itself runs, taking as input (a) the
  results of move analysis and (b) the regions computed by the region
  checker. This allows us to figure out which loans are still in scope
  at any particular point.

