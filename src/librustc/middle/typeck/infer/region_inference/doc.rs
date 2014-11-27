// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Region inference module.
//!
//! # Terminology
//!
//! Note that we use the terms region and lifetime interchangeably,
//! though the term `lifetime` is preferred.
//!
//! # Introduction
//!
//! Region inference uses a somewhat more involved algorithm than type
//! inference.  It is not the most efficient thing ever written though it
//! seems to work well enough in practice (famous last words).  The reason
//! that we use a different algorithm is because, unlike with types, it is
//! impractical to hand-annotate with regions (in some cases, there aren't
//! even the requisite syntactic forms).  So we have to get it right, and
//! it's worth spending more time on a more involved analysis.  Moreover,
//! regions are a simpler case than types: they don't have aggregate
//! structure, for example.
//!
//! Unlike normal type inference, which is similar in spirit to H-M and thus
//! works progressively, the region type inference works by accumulating
//! constraints over the course of a function.  Finally, at the end of
//! processing a function, we process and solve the constraints all at
//! once.
//!
//! The constraints are always of one of three possible forms:
//!
//! - ConstrainVarSubVar(R_i, R_j) states that region variable R_i
//!   must be a subregion of R_j
//! - ConstrainRegSubVar(R, R_i) states that the concrete region R
//!   (which must not be a variable) must be a subregion of the varibale R_i
//! - ConstrainVarSubReg(R_i, R) is the inverse
//!
//! # Building up the constraints
//!
//! Variables and constraints are created using the following methods:
//!
//! - `new_region_var()` creates a new, unconstrained region variable;
//! - `make_subregion(R_i, R_j)` states that R_i is a subregion of R_j
//! - `lub_regions(R_i, R_j) -> R_k` returns a region R_k which is
//!   the smallest region that is greater than both R_i and R_j
//! - `glb_regions(R_i, R_j) -> R_k` returns a region R_k which is
//!   the greatest region that is smaller than both R_i and R_j
//!
//! The actual region resolution algorithm is not entirely
//! obvious, though it is also not overly complex.
//!
//! ## Snapshotting
//!
//! It is also permitted to try (and rollback) changes to the graph.  This
//! is done by invoking `start_snapshot()`, which returns a value.  Then
//! later you can call `rollback_to()` which undoes the work.
//! Alternatively, you can call `commit()` which ends all snapshots.
//! Snapshots can be recursive---so you can start a snapshot when another
//! is in progress, but only the root snapshot can "commit".
//!
//! # Resolving constraints
//!
//! The constraint resolution algorithm is not super complex but also not
//! entirely obvious.  Here I describe the problem somewhat abstractly,
//! then describe how the current code works.  There may be other, smarter
//! ways of doing this with which I am unfamiliar and can't be bothered to
//! research at the moment. - NDM
//!
//! ## The problem
//!
//! Basically our input is a directed graph where nodes can be divided
//! into two categories: region variables and concrete regions.  Each edge
//! `R -> S` in the graph represents a constraint that the region `R` is a
//! subregion of the region `S`.
//!
//! Region variable nodes can have arbitrary degree.  There is one region
//! variable node per region variable.
//!
//! Each concrete region node is associated with some, well, concrete
//! region: e.g., a free lifetime, or the region for a particular scope.
//! Note that there may be more than one concrete region node for a
//! particular region value.  Moreover, because of how the graph is built,
//! we know that all concrete region nodes have either in-degree 1 or
//! out-degree 1.
//!
//! Before resolution begins, we build up the constraints in a hashmap
//! that maps `Constraint` keys to spans.  During resolution, we construct
//! the actual `Graph` structure that we describe here.
//!
//! ## Our current algorithm
//!
//! We divide region variables into two groups: Expanding and Contracting.
//! Expanding region variables are those that have a concrete region
//! predecessor (direct or indirect).  Contracting region variables are
//! all others.
//!
//! We first resolve the values of Expanding region variables and then
//! process Contracting ones.  We currently use an iterative, fixed-point
//! procedure (but read on, I believe this could be replaced with a linear
//! walk).  Basically we iterate over the edges in the graph, ensuring
//! that, if the source of the edge has a value, then this value is a
//! subregion of the target value.  If the target does not yet have a
//! value, it takes the value from the source.  If the target already had
//! a value, then the resulting value is Least Upper Bound of the old and
//! new values. When we are done, each Expanding node will have the
//! smallest region that it could possibly have and still satisfy the
//! constraints.
//!
//! We next process the Contracting nodes.  Here we again iterate over the
//! edges, only this time we move values from target to source (if the
//! source is a Contracting node).  For each contracting node, we compute
//! its value as the GLB of all its successors.  Basically contracting
//! nodes ensure that there is overlap between their successors; we will
//! ultimately infer the largest overlap possible.
//!
//! # The Region Hierarchy
//!
//! ## Without closures
//!
//! Let's first consider the region hierarchy without thinking about
//! closures, because they add a lot of complications. The region
//! hierarchy *basically* mirrors the lexical structure of the code.
//! There is a region for every piece of 'evaluation' that occurs, meaning
//! every expression, block, and pattern (patterns are considered to
//! "execute" by testing the value they are applied to and creating any
//! relevant bindings).  So, for example:
//!
//!     fn foo(x: int, y: int) { // -+
//!     //  +------------+       //  |
//!     //  |      +-----+       //  |
//!     //  |  +-+ +-+ +-+       //  |
//!     //  |  | | | | | |       //  |
//!     //  v  v v v v v v       //  |
//!         let z = x + y;       //  |
//!         ...                  //  |
//!     }                        // -+
//!
//!     fn bar() { ... }
//!
//! In this example, there is a region for the fn body block as a whole,
//! and then a subregion for the declaration of the local variable.
//! Within that, there are sublifetimes for the assignment pattern and
//! also the expression `x + y`. The expression itself has sublifetimes
//! for evaluating `x` and `y`.
//!
//! ## Function calls
//!
//! Function calls are a bit tricky. I will describe how we handle them
//! *now* and then a bit about how we can improve them (Issue #6268).
//!
//! Consider a function call like `func(expr1, expr2)`, where `func`,
//! `arg1`, and `arg2` are all arbitrary expressions. Currently,
//! we construct a region hierarchy like:
//!
//!     +----------------+
//!     |                |
//!     +--+ +---+  +---+|
//!     v  v v   v  v   vv
//!     func(expr1, expr2)
//!
//! Here you can see that the call as a whole has a region and the
//! function plus arguments are subregions of that. As a side-effect of
//! this, we get a lot of spurious errors around nested calls, in
//! particular when combined with `&mut` functions. For example, a call
//! like this one
//!
//!     self.foo(self.bar())
//!
//! where both `foo` and `bar` are `&mut self` functions will always yield
//! an error.
//!
//! Here is a more involved example (which is safe) so we can see what's
//! going on:
//!
//!     struct Foo { f: uint, g: uint }
//!     ...
//!     fn add(p: &mut uint, v: uint) {
//!         *p += v;
//!     }
//!     ...
//!     fn inc(p: &mut uint) -> uint {
//!         *p += 1; *p
//!     }
//!     fn weird() {
//!         let mut x: Box<Foo> = box Foo { ... };
//!         'a: add(&mut (*x).f,
//!                 'b: inc(&mut (*x).f)) // (..)
//!     }
//!
//! The important part is the line marked `(..)` which contains a call to
//! `add()`. The first argument is a mutable borrow of the field `f`.  The
//! second argument also borrows the field `f`. Now, in the current borrow
//! checker, the first borrow is given the lifetime of the call to
//! `add()`, `'a`.  The second borrow is given the lifetime of `'b` of the
//! call to `inc()`. Because `'b` is considered to be a sublifetime of
//! `'a`, an error is reported since there are two co-existing mutable
//! borrows of the same data.
//!
//! However, if we were to examine the lifetimes a bit more carefully, we
//! can see that this error is unnecessary. Let's examine the lifetimes
//! involved with `'a` in detail. We'll break apart all the steps involved
//! in a call expression:
//!
//!     'a: {
//!         'a_arg1: let a_temp1: ... = add;
//!         'a_arg2: let a_temp2: &'a mut uint = &'a mut (*x).f;
//!         'a_arg3: let a_temp3: uint = {
//!             let b_temp1: ... = inc;
//!             let b_temp2: &'b = &'b mut (*x).f;
//!             'b_call: b_temp1(b_temp2)
//!         };
//!         'a_call: a_temp1(a_temp2, a_temp3) // (**)
//!     }
//!
//! Here we see that the lifetime `'a` includes a number of substatements.
//! In particular, there is this lifetime I've called `'a_call` that
//! corresponds to the *actual execution of the function `add()`*, after
//! all arguments have been evaluated. There is a corresponding lifetime
//! `'b_call` for the execution of `inc()`. If we wanted to be precise
//! about it, the lifetime of the two borrows should be `'a_call` and
//! `'b_call` respectively, since the references that were created
//! will not be dereferenced except during the execution itself.
//!
//! However, this model by itself is not sound. The reason is that
//! while the two references that are created will never be used
//! simultaneously, it is still true that the first reference is
//! *created* before the second argument is evaluated, and so even though
//! it will not be *dereferenced* during the evaluation of the second
//! argument, it can still be *invalidated* by that evaluation. Consider
//! this similar but unsound example:
//!
//!     struct Foo { f: uint, g: uint }
//!     ...
//!     fn add(p: &mut uint, v: uint) {
//!         *p += v;
//!     }
//!     ...
//!     fn consume(x: Box<Foo>) -> uint {
//!         x.f + x.g
//!     }
//!     fn weird() {
//!         let mut x: Box<Foo> = box Foo { ... };
//!         'a: add(&mut (*x).f, consume(x)) // (..)
//!     }
//!
//! In this case, the second argument to `add` actually consumes `x`, thus
//! invalidating the first argument.
//!
//! So, for now, we exclude the `call` lifetimes from our model.
//! Eventually I would like to include them, but we will have to make the
//! borrow checker handle this situation correctly. In particular, if
//! there is a reference created whose lifetime does not enclose
//! the borrow expression, we must issue sufficient restrictions to ensure
//! that the pointee remains valid.
//!
//! ## Adding closures
//!
//! The other significant complication to the region hierarchy is
//! closures. I will describe here how closures should work, though some
//! of the work to implement this model is ongoing at the time of this
//! writing.
//!
//! The body of closures are type-checked along with the function that
//! creates them. However, unlike other expressions that appear within the
//! function body, it is not entirely obvious when a closure body executes
//! with respect to the other expressions. This is because the closure
//! body will execute whenever the closure is called; however, we can
//! never know precisely when the closure will be called, especially
//! without some sort of alias analysis.
//!
//! However, we can place some sort of limits on when the closure
//! executes.  In particular, the type of every closure `fn:'r K` includes
//! a region bound `'r`. This bound indicates the maximum lifetime of that
//! closure; once we exit that region, the closure cannot be called
//! anymore. Therefore, we say that the lifetime of the closure body is a
//! sublifetime of the closure bound, but the closure body itself is unordered
//! with respect to other parts of the code.
//!
//! For example, consider the following fragment of code:
//!
//!     'a: {
//!          let closure: fn:'a() = || 'b: {
//!              'c: ...
//!          };
//!          'd: ...
//!     }
//!
//! Here we have four lifetimes, `'a`, `'b`, `'c`, and `'d`. The closure
//! `closure` is bounded by the lifetime `'a`. The lifetime `'b` is the
//! lifetime of the closure body, and `'c` is some statement within the
//! closure body. Finally, `'d` is a statement within the outer block that
//! created the closure.
//!
//! We can say that the closure body `'b` is a sublifetime of `'a` due to
//! the closure bound. By the usual lexical scoping conventions, the
//! statement `'c` is clearly a sublifetime of `'b`, and `'d` is a
//! sublifetime of `'d`. However, there is no ordering between `'c` and
//! `'d` per se (this kind of ordering between statements is actually only
//! an issue for dataflow; passes like the borrow checker must assume that
//! closures could execute at any time from the moment they are created
//! until they go out of scope).
//!
//! ### Complications due to closure bound inference
//!
//! There is only one problem with the above model: in general, we do not
//! actually *know* the closure bounds during region inference! In fact,
//! closure bounds are almost always region variables! This is very tricky
//! because the inference system implicitly assumes that we can do things
//! like compute the LUB of two scoped lifetimes without needing to know
//! the values of any variables.
//!
//! Here is an example to illustrate the problem:
//!
//!     fn identify<T>(x: T) -> T { x }
//!
//!     fn foo() { // 'foo is the function body
//!       'a: {
//!            let closure = identity(|| 'b: {
//!                'c: ...
//!            });
//!            'd: closure();
//!       }
//!       'e: ...;
//!     }
//!
//! In this example, the closure bound is not explicit. At compile time,
//! we will create a region variable (let's call it `V0`) to represent the
//! closure bound.
//!
//! The primary difficulty arises during the constraint propagation phase.
//! Imagine there is some variable with incoming edges from `'c` and `'d`.
//! This means that the value of the variable must be `LUB('c,
//! 'd)`. However, without knowing what the closure bound `V0` is, we
//! can't compute the LUB of `'c` and `'d`! Any we don't know the closure
//! bound until inference is done.
//!
//! The solution is to rely on the fixed point nature of inference.
//! Basically, when we must compute `LUB('c, 'd)`, we just use the current
//! value for `V0` as the closure's bound. If `V0`'s binding should
//! change, then we will do another round of inference, and the result of
//! `LUB('c, 'd)` will change.
//!
//! One minor implication of this is that the graph does not in fact track
//! the full set of dependencies between edges. We cannot easily know
//! whether the result of a LUB computation will change, since there may
//! be indirect dependencies on other variables that are not reflected on
//! the graph. Therefore, we must *always* iterate over all edges when
//! doing the fixed point calculation, not just those adjacent to nodes
//! whose values have changed.
//!
//! Were it not for this requirement, we could in fact avoid fixed-point
//! iteration altogether. In that universe, we could instead first
//! identify and remove strongly connected components (SCC) in the graph.
//! Note that such components must consist solely of region variables; all
//! of these variables can effectively be unified into a single variable.
//! Once SCCs are removed, we are left with a DAG.  At this point, we
//! could walk the DAG in topological order once to compute the expanding
//! nodes, and again in reverse topological order to compute the
//! contracting nodes. However, as I said, this does not work given the
//! current treatment of closure bounds, but perhaps in the future we can
//! address this problem somehow and make region inference somewhat more
//! efficient. Note that this is solely a matter of performance, not
//! expressiveness.
//!
//! ### Skolemization
//!
//! For a discussion on skolemization and higher-ranked subtyping, please
//! see the module `middle::typeck::infer::higher_ranked::doc`.
