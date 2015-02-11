- Feature Name: (none for the bulk of RFC); unsafe_no_drop_flag
- Start Date: 2014-09-24
- RFC PR: [rust-lang/rfcs#320](https://github.com/rust-lang/rfcs/pull/320)
- Rust Issue: [rust-lang/rust#5016](https://github.com/rust-lang/rust/issues/5016)

# Summary

Remove drop flags from values implementing `Drop`, and remove
automatic memory zeroing associated with dropping values.

Keep dynamic drop semantics, by having each function maintain a
(potentially empty) set of auto-injected boolean flags for the drop
obligations for the function that need to be tracked dynamically
(which we will call "dynamic drop obligations").

# Motivation

Currently, implementing `Drop` on a struct (or enum) injects a hidden
bit, known as the "drop-flag", into the struct (and likewise, each of
the enum variants).  The drop-flag, in tandem with Rust's implicit
zeroing of dropped values, tracks whether a value has already been
moved to another owner or been dropped.  (See the ["How dynamic drop
semantics works"](#how-dynamic-drop-semantics-works) appendix for more
details if you are unfamiliar with this part of Rust's current
implementation.)

However, the above implementation is sub-optimal; problems include:

 * Most important: implicit memory zeroing is a hidden cost that today
   all Rust programs pay, in both execution time and code size.
   With the removal of the drop flag, we can remove implicit memory
   zeroing (or at least revisit its utility -- there may be other
   motivations for implicit memory zeroing, e.g. to try to keep secret
   data from being exposed to unsafe code).

 * Hidden bits are bad: Users coming from a C/C++ background
   expect `struct Foo { x: u32, y: u32 }` to occupy 8 bytes, but if
   `Foo` implements `Drop`, the hidden drop flag will cause it to
   double in size (16 bytes).
   See the [Program illustrating semantic impact of hidden drop flag]
   appendix for a concrete illustration.  Note that when `Foo`
   implements `Drop`, each instance of `Foo` carries a drop-flag, even
   in contexts like a `Vec<Foo>` where a program
   cannot actually move individual values out of the collection.
   Thus, the amount of extra memory being used by drop-flags is not
   bounded by program stack growth; the memory wastage is strewn
   throughout the heap.

An earlier RFC (the withdrawn [RFC PR #210]) suggested resolving this
problem by switching from a dynamic drop semantics to a "static drop
semantics", which was defined in that RFC as one that performs drop of
certain values earlier to ensure that the set of drop-obligations does
not differ at any control-flow merge point, i.e. to ensure that the
set of values to drop is statically known at compile-time.

[RFC PR #210]: https://github.com/rust-lang/rfcs/pull/210

However, discussion on the [RFC PR #210] comment thread pointed out
its policy for inserting early drops into the code is non-intuitive
(in other words, that the drop policy should either be more
aggressive, a la [RFC PR #239], or should stay with the dynamic drop
status quo). Also, the mitigating mechanisms proposed by that RFC
(`NoisyDrop`/`QuietDrop`) were deemed unacceptable.

[RFC PR #239]: https://github.com/rust-lang/rfcs/pull/239

So, static drop semantics are a non-starter.  Luckily, the above
strategy is not the only way to implement dynamic drop semantics.
Rather than requiring that the set of drop-obligations be the same at
every control-flow merge point, we can do a intra-procedural static
analysis to identify the set of drop-obligations that differ at any
merge point, and then inject a set of stack-local boolean-valued
drop-flags that dynamically track them.  That strategy is what this
RFC is describing.

The expected outcomes are as follows:

 * We remove the drop-flags from all structs/enums that implement
   `Drop`. (There are still the injected stack-local drop flags, but
   those should be cheaper to inject and maintain.)

 * Since invoking drop code is now handled by the stack-local drop
   flags and we have no more drop-flags on the values themselves,
   we can (and will) remove memory zeroing.

 * Libraries currently relying on drop doing memory zeroing (i.e.
   libraries that check whether content is zero to decide whether its
   `fn drop` has been invoked will need to be revised, since we will
   not have implicit memory zeroing anymore.

 * In the common case, most libraries using `Drop` will not need to
   change at all from today, apart from the caveat in the previous
   bullet.
 
# Detailed design


## Drop obligations

No struct or enum has an implicit drop-flag.  When a local variable is
initialized, that establishes a set of "drop obligations": a set of
structural paths (e.g. a local `a`, or a path to a field `b.f.y`) that
need to be dropped (or moved away to a new owner).

The drop obligations for a local variable `x` of struct-type `T` are
computed from analyzing the structure of `T`.  If `T` itself
implements `Drop`, then `x` is a drop obligation.  If `T` does not
implement `Drop`, then the set of drop obligations is the union of the
drop obligations of the fields of `T`.

When a path is moved to a new location, or consumed by a function call,
or when control flow reaches the end of its owner's lexical scope,
the path is removed from the set of drop obligations.

At control-flow merge points, e.g. nodes that have predecessor nodes
P_1, P_2, ..., P_k with drop obligation sets S_1, S_2, ... S_k, we

 * First identify the set of drop obligations that differ between the
   predecessor nodes, i.e. the set:

     `(S_1 | S_2 | ... | S_k) \ (S_1 & S_2 & ... & S_k)`

   where `|` denotes set-union, `&` denotes set-intersection, 
   `\` denotes set-difference.  These are the dynamic drop obligations
   induced by this merge point.  Note that if `S_1 = S_2 = ... = S_k`,
   the above set is empty.

 * The set of drop obligations for the merge point itself is the
   union of the drop-obligations from all predecessor points in
   the control flow, i.e. `(S_1 | S_2 | ... | S_k)` in the
   above notation.

   (One could also just use the intersection here; it actually makes
   no difference to the static analysis, since all of the elements of
   the difference

     `(S_1 | S_2 | ... | S_k) \ (S_1 & S_2 & ... & S_k)`

   have already been added to the set of dynamic drop obligations.
   But the overall code transformation is clearer if one keeps
   the dynamic drop obligations in the set of drop obligations.)

## Stack-local drop flags

For every dynamic drop obligation induced by a merge point, the compiler
is responsible for ensure that its drop code is run at some point.
If necessary, it will inject and maintain boolean flag analogous to
```rust
enum NeedsDropFlag { NeedsLocalDrop, DoNotDrop }
```

Some compiler analysis may be able to identify dynamic drop
obligations that do not actually need to be tracked.  Therefore, we do
not specify the precise set of boolean flags that are injected.

## Example of code with dynamic drop obligations


The function `f2` below was copied from the static drop [RFC PR #210];
it has differing sets of drop obligations at a merge point,
necessitating a potential injection of a `NeedsDropFlag`.

```rust
fn f2() {

    // At the outset, the set of drop obligations is
    // just the set of moved input parameters (empty
    // in this case).

    //                                      DROP OBLIGATIONS
    //                                  ------------------------
    //                                  {  }
    let pDD : Pair<D,D> = ...;
    pDD.x = ...;
    //                                  {pDD.x}
    pDD.y = ...;
    //                                  {pDD.x, pDD.y}
    let pDS : Pair<D,S> = ...;
    //                                  {pDD.x, pDD.y, pDS.x}
    let some_d : Option<D>;
    //                                  {pDD.x, pDD.y, pDS.x}
    if test() {
        //                                  {pDD.x, pDD.y, pDS.x}
        {
            let temp = xform(pDD.y);
            //                              {pDD.x,        pDS.x, temp}
            some_d = Some(temp);
            //                              {pDD.x,        pDS.x, temp, some_d}
        } // END OF SCOPE for `temp`
        //                                  {pDD.x,        pDS.x, some_d}

        // MERGE POINT PREDECESSOR 1

    } else {
        {
            //                              {pDD.x, pDD.y, pDS.x}
            let z = D;
            //                              {pDD.x, pDD.y, pDS.x, z}

            // This drops `pDD.y` before
            // moving `pDD.x` there.
            pDD.y = pDD.x;

            //                              {       pDD.y, pDS.x, z}
            some_d = None;
            //                              {       pDD.y, pDS.x, z, some_d}
        } // END OF SCOPE for `z`
        //                                  {       pDD.y, pDS.x, some_d}

        // MERGE POINT PREDECESSOR 2

    }

    // MERGE POINT: set of drop obligations do not
    // match on all incoming control-flow paths.
    //
    // Predecessor 1 has drop obligations
    // {pDD.x,        pDS.x, some_d}
    // and Predecessor 2 has drop obligations
    // {       pDD.y, pDS.x, some_d}.
    //
    // Therefore, this merge point implies that
    // {pDD.x, pDD.y} are dynamic drop obligations,
    // while {pDS.x, some_d} are potentially still
    // resolvable statically (and thus may not need
    // associated boolean flags).

    // The resulting drop obligations are the following:

    //                                  {pDD.x, pDD.y, pDS.x, some_d}.

    // (... some code that does not change drop obligations ...)

    //                                  {pDD.x, pDD.y, pDS.x, some_d}.

    // END OF SCOPE for `pDD`, `pDS`, `some_d`
}
```

After the static analysis has identified all of the dynamic drop
obligations, code is injected to maintain the stack-local drop flags
and to do any necessary drops at the appropriate points.
Below is the updated `fn f2` with an approximation of the injected code.

Note: we say "approximation", because one does need to ensure that the
drop flags are updated in a manner that is compatible with potential
task `fail!`/`panic!`, because stack unwinding must be informed which
state needs to be dropped; i.e. you need to initialize `_pDD_dot_x`
before you start to evaluate a fallible expression to initialize
`pDD.y`.


```rust
fn f2_rewritten() {

    // At the outset, the set of drop obligations is
    // just the set of moved input parameters (empty
    // in this case).

    //                                      DROP OBLIGATIONS
    //                                  ------------------------
    //                                  {  }
    let _drop_pDD_dot_x : NeedsDropFlag;
    let _drop_pDD_dot_y : NeedsDropFlag;

    _drop_pDD_dot_x = DoNotDrop;
    _drop_pDD_dot_y = DoNotDrop;

    let pDD : Pair<D,D>;
    pDD.x = ...;
    _drop_pDD_dot_x = NeedsLocalDrop;
    pDD.y = ...;
    _drop_pDD_dot_y = NeedsLocalDrop;

    //                                  {pDD.x, pDD.y}
    let pDS : Pair<D,S> = ...;
    //                                  {pDD.x, pDD.y, pDS.x}
    let some_d : Option<D>;
    //                                  {pDD.x, pDD.y, pDS.x}
    if test() {
        //                                  {pDD.x, pDD.y, pDS.x}
        {
            _drop_pDD_dot_y = DoNotDrop;
            let temp = xform(pDD.y);
            //                              {pDD.x,        pDS.x, temp}
            some_d = Some(temp);
            //                              {pDD.x,        pDS.x, temp, some_d}
        } // END OF SCOPE for `temp`
        //                                  {pDD.x,        pDS.x, some_d}

        // MERGE POINT PREDECESSOR 1

    } else {
        {
            //                              {pDD.x, pDD.y, pDS.x}
            let z = D;
            //                              {pDD.x, pDD.y, pDS.x, z}

            // This drops `pDD.y` before
            // moving `pDD.x` there.
            _drop_pDD_dot_x = DoNotDrop;
            pDD.y = pDD.x;

            //                              {       pDD.y, pDS.x, z}
            some_d = None;
            //                              {       pDD.y, pDS.x, z, some_d}
        } // END OF SCOPE for `z`
        //                                  {       pDD.y, pDS.x, some_d}

        // MERGE POINT PREDECESSOR 2

    }

    // MERGE POINT: set of drop obligations do not
    // match on all incoming control-flow paths.
    //
    // Predecessor 1 has drop obligations
    // {pDD.x,        pDS.x, some_d}
    // and Predecessor 2 has drop obligations
    // {       pDD.y, pDS.x, some_d}.
    //
    // Therefore, this merge point implies that
    // {pDD.x, pDD.y} are dynamic drop obligations,
    // while {pDS.x, some_d} are potentially still
    // resolvable statically (and thus may not need
    // associated boolean flags).

    // The resulting drop obligations are the following:

    //                                  {pDD.x, pDD.y, pDS.x, some_d}.

    // (... some code that does not change drop obligations ...)

    //                                  {pDD.x, pDD.y, pDS.x, some_d}.

    // END OF SCOPE for `pDD`, `pDS`, `some_d`

    // rustc-inserted code (not legal Rust, since `pDD.x` and `pDD.y`
    // are inaccessible).

    if _drop_pDD_dot_x { mem::drop(pDD.x); }
    if _drop_pDD_dot_y { mem::drop(pDD.y); }
}
```

Note that in a snippet like
```rust
       _drop_pDD_dot_y = DoNotDrop;
       let temp = xform(pDD.y);
```
this is okay, in part because the evaluating the identifier `xform` is
infallible.  If instead it were something like:
```rust
       _drop_pDD_dot_y = DoNotDrop;
       let temp = lookup_closure()(pDD.y);
```
then that would not be correct, because we need to set
`_drop_pDD_dot_y` to `DoNotDrop` after the `lookup_closure()`
invocation.

It may probably be more intellectually honest to write the transformation like:
```rust
       let temp = lookup_closure()({ _drop_pDD_dot_y = DoNotDrop; pDD.y });
```


## Control-flow sensitivity

Note that the dynamic drop obligations are based on a control-flow
analysis, *not* just the lexical nesting structure of the code.

In particular: If control flow splits at a point like an if-expression,
but the two arms never meet, then they can have completely
sets of drop obligations.

This is important, since in coding patterns like loops, one
often sees different sets of drop obligations prior to a `break`
compared to a point where the loop repeats, such as a `continue`
or the end of a `loop` block.

```rust
    // At the outset, the set of drop obligations is
    // just the set of moved input parameters (empty
    // in this case).

    //                                      DROP OBLIGATIONS
    //                                  ------------------------
    //                                  {  }
    let mut pDD : Pair<D,D> = mk_dd();
    let mut maybe_set : D;

    //                                  {         pDD.x, pDD.y }
    'a: loop {
        // MERGE POINT

        //                                  {     pDD.x, pDD.y }
        if test() {
            //                                  { pDD.x, pDD.y }
            consume(pDD.x);
            //                                  {        pDD.y }
            break 'a;
        }
        // *not* merge point (only one path, the else branch, flows here)

        //                                  {     pDD.x, pDD.y }

        // never falls through; must merge with 'a loop.
    }

    // RESUME POINT: break 'a above flows here

    //                                  {                pDD.y }

    // This is the point immediately preceding `'b: loop`; (1.) below.

    'b: loop {
        // MERGE POINT
        //
        // There are *three* incoming paths: (1.) the statement
        // preceding `'b: loop`, (2.) the `continue 'b;` below, and
        // (3.) the end of the loop's block below.  The drop
        // obligation for `maybe_set` originates from (3.).

        //                                  {            pDD.y, maybe_set }

        consume(pDD.y);

        //                                  {                 , maybe_set }

        if test() {
            //                                  {             , maybe_set }
            pDD.x = mk_d();
            //                                  { pDD.x       , maybe_set }
            break 'b;
        }

        // *not* merge point (only one path flows here)

        //                                  {                 , maybe_set }

        if test() {
            //                                  {             , maybe_set }
            pDD.y = mk_d();

            // This is (2.) referenced above.   {        pDD.y, maybe_set }
            continue 'b;
        }
        // *not* merge point (only one path flows here)

        //                                  {                 , maybe_set }

        pDD.y = mk_d();
        // This is (3.) referenced above.   {            pDD.y, maybe_set }

        maybe_set = mk_d();
        g(&maybe_set);

        // This is (3.) referenced above.   {            pDD.y, maybe_set }
    }

    // RESUME POINT: break 'b above flows here

    //                                  {         pDD.x       , maybe_set }

    // when we hit the end of the scope of `maybe_set`;
    // check its stack-local flag.
```

Likewise, a `return` statement represents another control flow jump,
to the end of the function.

## Remove implicit memory zeroing

With the above in place, the remainder is relatively trivial.
The compiler can be revised to no longer inject a drop flag into
structs and enums that implement `Drop`, and likewise memory zeroing can
be removed.

Beyond that, the libraries will obviously need to be audited for
dependence on implicit memory zeroing.

# Drawbacks

The only reasons not do this are:

 1. Some hypothetical reason to *continue* doing implicit memory zeroing, or

 2. We want to abandon dynamic drop semantics.

At this point Felix thinks the Rust community has made a strong
argument in favor of keeping dynamic drop semantics.

# Alternatives

* Static drop semantics [RFC PR #210] has been referenced frequently
  in this document.

* Eager drops [RFC PR #239] is the more aggressive semantics that
  would drop values immediately after their final use.  This would
  probably invalidate a number of RAII style coding patterns.

# Optional Extensions

## A lint identifying dynamic drop obligations

Add a lint (set by default to `allow`) that reports potential dynamic
drop obligations, so that end-user code can opt-in to having them
reported.  The expected benefits of this are:

 1. developers may have intended for a value to be moved elsewhere on
    all paths within a function, and,

 2. developers may want to know about how many boolean dynamic drop
    flags are potentially being injected into their code.

# Unresolved questions

## How to handle moves out of `array[index_expr]`

Niko pointed out to me today that my prototype was not addressing
moves out of `array[index_expr]` properly.  I was assuming
that we would just make such an expression illegal (or that they
should already be illegal).

But they are not already illegal, and above assumption that we
would make it illegal should have been explicit.  That, or we
should address the problem in some other way.

To make this concrete, here is some code that runs today:

```rust
#[deriving(Show)]
struct AnnounceDrop { name: &'static str }

impl Drop for AnnounceDrop {
    fn drop(&mut self) { println!("dropping {}", self.name); }
}

fn foo<A>(a: [A, ..3], i: uint) -> A {
    a[i]
}

fn main() {
    let a = [AnnounceDrop { name: "fst" },
             AnnounceDrop { name: "snd" },
             AnnounceDrop { name: "thd" }];
    let r = foo(a, 1);
    println!("foo returned {}", r);
}
```

This prints:
```
dropping fst
dropping thd
foo returned AnnounceDrop { name: snd }
dropping snd
```

because it first moves the entire array into `foo`, and then `foo`
returns the second element, but still needs to drop the rest of the
array.

Embedded drop flags and zeroing support this seamlessly, of course.
But the whole point of this RFC is to get rid of the embedded
per-value drop-flags.

If we want to continue supporting moving out of `a[i]` (and we
probably do, I have been converted on this point), then the drop flag
needs to handle this case.  Our current thinking is that we can
support it by using a single *`uint`* flag (as opposed to the booleans
used elsewhere) for such array that has been moved out of.  The `uint`
flag represents "drop all elements from the array *except* for the one
listed in the flag."  (If it is only moved out of on one branch and
not another, then we would either use an `Option<uint>`, or still use
`uint` and just represent unmoved case via some value that is not
valid index, such as the length of the array).

## Should we keep `#[unsafe_no_drop_flag]` ?

Currently there is an `unsafe_no_drop_flag` attribute that is used to
indicate that no drop flag should be associated with a struct/enum,
and instead the user-written drop code will be run multiple times (and
thus must internally guard itself from its own side-effects; e.g. do
not attempt to free the backing buffer for a `Vec` more than once, by
tracking within the `Vec` itself if the buffer was previously freed).

The "obvious" thing to do is to remove `unsafe_no_drop_flag`, since
the per-value drop flag is going away.  However, we *could* keep the
attribute, and just repurpose its meaning to instead mean the
following: *Never* inject a dynamic stack-local drop-flag for this
value.  Just run the drop code multiple times, just like today.

In any case, since the semantics of this attribute are unstable, we
will feature-gate it (with feature name `unsafe_no_drop_flag`).

# Appendices

## How dynamic drop semantics works

(This section is just presenting background information on the
semantics of `drop` and the drop-flag as it works in Rust today; it
does not contain any discussion of the changes being proposed by this
RFC.)

A struct or enum implementing `Drop` will have its drop-flag
automatically set to a non-zero value when it is constructed.  When
attempting to drop the struct or enum (i.e. when control reaches the
end of the lexical scope of its owner), the injected glue code will
only execute its associated `fn drop` if its drop-flag is non-zero.

In addition, the compiler injects code to ensure that when a value is
moved to a new location in memory or dropped, then the original memory
is entirely zeroed.

A struct/enum definition implementing `Drop` can be tagged with the
attribute `#[unsafe_no_drop_flag]`.  When so tagged, the struct/enum
will not have a hidden drop flag embedded within it. In this case, the
injected glue code will execute the associated glue code
unconditionally, even though the struct/enum value may have been moved
to a new location in memory or dropped (in either case, the memory
representing the value will have been zeroed).

The above has a number of implications:

 * A program can manually cause the drop code associated with a value
   to be skipped by first zeroing out its memory.

 * A `Drop` implementation for a struct tagged with `unsafe_no_drop_flag`
   must assume that it will be called more than once.  (However, every
   call to `drop` after the first will be given zeroed memory.)

### Program illustrating semantic impact of hidden drop flag

```rust
#![feature(macro_rules)]

use std::fmt;
use std::mem;

#[deriving(Clone,Show)]
struct S {  name: &'static str }

#[deriving(Clone,Show)]
struct Df { name: &'static str }

#[deriving(Clone,Show)]
struct Pair<X,Y>{ x: X, y: Y }

static mut current_indent: uint = 0;

fn indent() -> String {
    String::from_char(unsafe { current_indent }, ' ')
}

impl Drop for Df {
    fn drop(&mut self) {
        println!("{}dropping Df {}", indent(), self.name)
    }
}

macro_rules! struct_Dn {
    ($Dn:ident) => {

        #[unsafe_no_drop_flag]
        #[deriving(Clone,Show)]
        struct $Dn { name: &'static str }

        impl Drop for $Dn {
            fn drop(&mut self) {
                if unsafe { (0,0) == mem::transmute::<_,(uint,uint)>(self.name) } {
                    println!("{}dropping already-zeroed {}",
                             indent(), stringify!($Dn));
                } else {
                    println!("{}dropping {} {}",
                             indent(), stringify!($Dn), self.name)
                }
            }
        }
    }
}

struct_Dn!(DnA)
struct_Dn!(DnB)
struct_Dn!(DnC)

fn take_and_pass<T:fmt::Show>(t: T) {
    println!("{}t-n-p took and will pass: {}", indent(), &t);
    unsafe { current_indent += 4; }
    take_and_drop(t);
    unsafe { current_indent -= 4; }
}

fn take_and_drop<T:fmt::Show>(t: T) {
    println!("{}t-n-d took and will drop: {}", indent(), &t);
}

fn xform(mut input: Df) -> Df {
    input.name = "transformed";
    input
}

fn foo(b: || -> bool) {
    let mut f1 = Df  { name: "f1" };
    let mut n2 = DnC { name: "n2" };
    let f3 = Df  { name: "f3" };
    let f4 = Df  { name: "f4" };
    let f5 = Df  { name: "f5" };
    let f6 = Df  { name: "f6" };
    let n7 = DnA { name: "n7" };
    let _fx = xform(f6);           // `f6` consumed by `xform`
    let _n9 = DnB { name: "n9" };
    let p = Pair { x: f4, y: f5 }; // `f4` and `f5` moved into `p`
    let _f10 = Df { name: "f10" };

    println!("foo scope start: {}", (&f3, &n7));
    unsafe { current_indent += 4; }
    if b() {
        take_and_pass(p.x); // `p.x` consumed by `take_and_pass`, which drops it
    }
    if b() {
        take_and_pass(n7); // `n7` consumed by `take_and_pass`, which drops it
    }
    
    // totally unsafe: manually zero the struct, including its drop flag.
    unsafe fn manually_zero<S>(s: &mut S) {
        let len = mem::size_of::<S>();
        let p : *mut u8 = mem::transmute(s);
        for i in range(0, len) {
            *p.offset(i as int) = 0;
        }
    }
    unsafe {
        manually_zero(&mut f1);
        manually_zero(&mut n2);
    }
    println!("foo scope end");
    unsafe { current_indent -= 4; }

    // here, we drop each local variable, in reverse order of declaration.
    // So we should see the following drop sequence:
    // drop(f10), printing "Df f10"
    // drop(p)
    //   ==> drop(p.y), printing "Df f5"
    //   ==> attempt to drop(and skip) already-dropped p.x, no-op
    // drop(_n9), printing "DnB n9"
    // drop(_fx), printing "Df transformed"
    // attempt to drop already-dropped n7, printing "already-zeroed DnA"
    // no drop of `f6` since it was consumed by `xform`
    // no drop of `f5` since it was moved into `p`
    // no drop of `f4` since it was moved into `p`
    // drop(f3), printing "f3"
    // attempt to drop manually-zeroed `n2`, printing "already-zeroed DnC"
    // attempt to drop manually-zeroed `f1`, no-op.
}

fn main() {
    foo(|| true);
}
```
