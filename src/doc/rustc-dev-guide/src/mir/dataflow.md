# Dataflow Analysis

If you work on the MIR, you will frequently come across various flavors of
[dataflow analysis][wiki]. For example, `rustc` uses dataflow to find
uninitialized variables, determine what variables are live across a generator
`yield` statement, and compute which `Place`s are borrowed at a given point in
the control-flow graph. Dataflow analysis is a fundamental concept in modern
compilers, and knowledge of the subject will be helpful to prospective
contributors.

However, this documentation is not a general introduction to dataflow analysis.
It is merely a description of the framework used to define these analyses in
`rustc`. It assumes that the reader is familiar with some basic terminology,
such as "transfer function", "fixpoint" and "lattice". If you're unfamiliar
with these terms, or if you want a quick refresher, [*Static Program Analysis*]
by Anders Møller and Michael I. Schwartzbach is an excellent, freely available
textbook.  For those who prefer audiovisual learning, the Goethe University
Frankfurt has published a series of short [youtube lectures][goethe] in English
that are very approachable.

## Defining a Dataflow Analysis

The interface for dataflow analyses is split into three traits. The first is
[`AnalysisDomain`], which must be implemented by *all* analyses. In addition to
the type of the dataflow state, this trait defines the initial value of that
state at entry to each block, as well as the direction of the analysis, either
forward or backward. The domain of your dataflow analysis must be a [lattice][]
(strictly speaking a join-semilattice) with a well-behaved `join` operator. See
documentation for the [`lattice`] module, as well as the [`JoinSemiLattice`]
trait, for more information.

You must then provide *either* a direct implementation of the [`Analysis`] trait
*or* an implementation of the proxy trait [`GenKillAnalysis`]. The latter is for
so-called ["gen-kill" problems], which have a simple class of transfer function
that can be applied very efficiently. Analyses whose domain is not a `BitSet`
of some index type, or whose transfer functions cannot be expressed through
"gen" and "kill" operations, must implement `Analysis` directly, and will run
slower as a result. All implementers of `GenKillAnalysis` also implement
`Analysis` automatically via a default `impl`.


```text
 AnalysisDomain
       ^
       |          | = has as a supertrait
       |          . = provides a default impl for
       |
   Analysis
     ^   ^
     |   .
     |   .
     |   .
 GenKillAnalysis

```

### Transfer Functions and Effects

The dataflow framework in `rustc` allows each statement inside a basic block as
well as the terminator to define its own transfer function. For brevity, these
individual transfer functions are known as "effects". Each effect is applied
successively in dataflow order, and together they define the transfer function
for the entire basic block. It's also possible to define an effect for
particular outgoing edges of some terminators (e.g.
[`apply_call_return_effect`] for the `success` edge of a `Call`
terminator). Collectively, these are known as per-edge effects.

The only meaningful difference (besides the "apply" prefix) between the methods
of the `GenKillAnalysis` trait and the `Analysis` trait is that an `Analysis`
has direct, mutable access to the dataflow state, whereas a `GenKillAnalysis`
only sees an implementer of the `GenKill` trait, which only allows the `gen`
and `kill` operations for mutation.

Observant readers of the documentation for these traits may notice that there
are actually *two* possible effects for each statement and terminator, the
"before" effect and the unprefixed (or "primary") effect. The "before" effects
are applied immediately before the unprefixed effect **regardless of whether
the analysis is backward or forward**. The vast majority of analyses should use
only the unprefixed effects: Having multiple effects for each statement makes
it difficult for consumers to know where they should be looking. However, the
"before" variants can be useful in some scenarios, such as when the effect of
the right-hand side of an assignment statement must be considered separately
from the left-hand side.

### Convergence

TODO

## Inspecting the Results of a Dataflow Analysis

Once you have constructed an analysis, you must pass it to an [`Engine`], which
is responsible for finding the steady-state solution to your dataflow problem.
You should use the [`into_engine`] method defined on the `Analysis` trait for
this, since it will use the more efficient `Engine::new_gen_kill` constructor
when possible.

Calling `iterate_to_fixpoint` on your `Engine` will return a `Results`, which
contains the dataflow state at fixpoint upon entry of each block. Once you have
a `Results`, you can can inspect the dataflow state at fixpoint at any point in
the CFG. If you only need the state at a few locations (e.g., each `Drop`
terminator) use a [`ResultsCursor`]. If you need the state at *every* location,
a [`ResultsVisitor`] will be more efficient.

```text
                         Analysis
                            |
                            | into_engine(…)
                            |
                          Engine
                            |
                            | iterate_to_fixpoint()
                            |
                         Results
                         /     \
 into_results_cursor(…) /       \  visit_with(…)
                       /         \
               ResultsCursor  ResultsVisitor
```

For example, the following code uses a [`ResultsVisitor`]...


```rust,ignore
// Assuming `MyVisitor` implements `ResultsVisitor<FlowState = MyAnalysis::Domain>`...
let my_visitor = MyVisitor::new();

// inspect the fixpoint state for every location within every block in RPO.
let results = MyAnalysis()
    .into_engine(tcx, body, def_id)
    .iterate_to_fixpoint()
    .visit_with(body, traversal::reverse_postorder(body), &mut my_visitor);
```

whereas this code uses [`ResultsCursor`]:

```rust,ignore
let mut results = MyAnalysis()
    .into_engine(tcx, body, def_id)
    .iterate_to_fixpoint()
    .into_results_cursor(body);

// Inspect the fixpoint state immediately before each `Drop` terminator.
for (bb, block) in body.basic_blocks().iter_enumerated() {
    if let TerminatorKind::Drop { .. } = block.terminator().kind {
        results.seek_before_primary_effect(body.terminator_loc(bb));
        let state = results.get();
        println!("state before drop: {:#?}", state);
    }
}
```

["gen-kill" problems]: https://en.wikipedia.org/wiki/Data-flow_analysis#Bit_vector_problems
[*Static Program Analysis*]: https://cs.au.dk/~amoeller/spa/
[`AnalysisDomain`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.AnalysisDomain.html
[`Analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.Analysis.html
[`GenKillAnalysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.GenKillAnalysis.html
[`JoinSemiLattice`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/lattice/trait.JoinSemiLattice.html
[`ResultsCursor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/struct.ResultsCursor.html
[`ResultsVisitor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.ResultsVisitor.html
[`apply_call_return_effect`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.Analysis.html#tymethod.apply_call_return_effect
[`into_engine`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/trait.Analysis.html#method.into_engine
[`lattice`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/dataflow/lattice/index.html
[goethe]: https://www.youtube.com/watch?v=NVBQSR_HdL0&list=PL_sGR8T76Y58l3Gck3ZwIIHLWEmXrOLV_&index=2
[lattice]: https://en.wikipedia.org/wiki/Lattice_(order)
[wiki]: https://en.wikipedia.org/wiki/Data-flow_analysis#Basic_principles
