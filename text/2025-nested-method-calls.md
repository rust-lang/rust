- Feature Name: nested_method_call
- Start Date: 2017-06-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/2025
- Rust Issue: https://github.com/rust-lang/rust/issues/44100

# Summary
[summary]: #summary

Enable "nested method calls" where the outer call is an `&mut self`
borrow, such as `vec.push(vec.len())` (where `vec: Vec<usize>`). This
is done by extending MIR with the concept of a **two-phase borrow**;
in this model, select `&mut` borrows are modified so that they begin
with a "reservation" phase and can later be "activated" into a full
mutable borrow. During the reservation phase, reads and shared borrows
of the borrowed data are permitted (but not mutation), as long as they
are confined to the reservation period. Once the mutable borrow is
activated, it acts like an ordinary mutable borrow.

Two-phase borrows in this RFC are only used when desugaring method
calls; this is intended as a conservative step. In the future, if
desired, the scheme could be extended to other syntactic forms, or
else subsumed as part of non-lexical lifetimes or some other
generalization of the lifetime system.

# Motivation
[motivation]: #motivation

The overriding goal here is that we want to accept nested method calls
where the outer call is an `&mut self` method, like
`vec.push(vec.len())`. This is a common limitation that beginners
stumble over and find confusing and which experienced users have as a
persistent annoyance. This makes it a natural target to eliminate as
part of the [2017 Roadmap][roadmap].

[roadmap]: https://github.com/rust-lang/rfcs/blob/master/text/1774-roadmap-2017.md

This problem has been extensively discussed on the internals
discussion board (e.g., [1][], [2][]), and a number of different
approaches to solving it have been proposed. This RFC itself is
intended to represent a "maximally minimal" approach, in the sense
that it tries to avoid making larger changes to the set of Rust code
that will be accepted, and instead focuses precisely on the
method-call form. It is compatible with the various alternatives, and
tries to leave room for future expansion in a variety of
directions. See the Alternatives section for more details.

[1]: https://internals.rust-lang.org/t/accepting-nested-method-calls-with-an-mut-self-receiver/4588
[2]: https://internals.rust-lang.org/t/blog-post-nested-method-calls-via-two-phase-borrowing/4886

## Why do we get an error in the first place?

You may wonder why this code isn't accepted in the first place. To see
why, consider what the (somewhat simplified) resulting MIR looks like:

[^simp]: This MIR is mildly simplified; the real MIR has multiple basic blocks to account for the possibility of panics.

```rust
/* 0 */ tmp0 = &'a mut vec;    // <-- mutable borrow starts here
/* 1 */ tmp1 = &'b vec;        // <-- shared borrow overlaps here
/* 2 */ tmp2 = Vec::len(tmp1);
/* 3 */ EndRegion('b);         // <-- shared borrow ends here
/* 3 */ Vec::push(tmp0, tmp2);
/* 5 */ EndRegion('a);         // <-- mutable borrow ends here
```

As you can see, we first take a mutable reference to `vec` for
`tmp0`. This "locks" `vec` from being accessed in any other way until
after the call to `Vec::push()`, but then we try to access it again
when calling `vec.len()`. Hence the error.

(In this MIR, I've included the `EndRegion` annotations that the
current MIR borrowck relies on. In most examples, I will elide them
unless they are needed to make a point. Also, in the future, when we
move to NLL, those statements will not be present, and regions will be
inferred based solely on where the references are *used*, but the
general idea remains the same.)

When you see the code desugared in that way, it should not surprise
you that there is in fact a real danger here for code to crash if we
just "turned off" this check (if we even could do such a thing). For
example, consider this rather artificial Rust program:

```rust
let mut v: Vec<String> = vec![format!("Hello, ")];
let s: String = format!("foo");
v[0].push_str({ v.push(s); "World!" });
//              ^^^^^^^^^ sneaky attempt to mutate `v`
```

This last line, if desugared into MIR, looks something like this;

```rust
// First evaluate `v[0]` to get a `&mut String`:
tmp0 = &mut v;
tmp1 = IndexMut::index_mut(tmp0, 0);
tmp2 = tmp1;

// Next, evaluate `{ v.push(s); "World!" }` block:
tmp3 = &mut v;
tmp4 = s;
Vec::push(tmp3, tmp4);
tmp5 = "World!";

// Finally, invoke `push_str`:
String::push_str(tmp2, tmp5);
```

The danger here lies in the fact that we evaluate `v[0]` into a
reference first, but this reference could well be invalidated by the
call to `Vec::push()` that occurs later on (which may resize the
vector and hence change the address of its elements). The Rust type
system naturally prevents this, however, because the first line (`tmp0
= &mut v`) borrows `v`, and that borrow lasts until the final call to
`push_str()`.

In fact, even when the receiver is just a local variable (e.g.,
`vec.push(vec.len())`) we have to be wary. We wouldn't want it to be
possible to give ownership of the receiver away in one of the
arguments: `vec.push({ send_to_another_thread(vec); ... })`. That
should still be an error of course.

(Naturally, these complex arguments that are blocks look really
artificial, but keep in mind that most of the time when this occurs in
practice, the argument is a method or fn call, and that could in
principle have arbitrary side-effects.)

### Introducing reservations

This RFC proposes extending MIR with the concept of a **two-phase
borrow**. These borrows are a variant of mutable borrows where the
value starts out as **reserved** and only becomes mutably borrowed
when the resulting reference is first used (which is called
**activating** the borrow). During the reservation phase before a
mutable borrow is activated, it acts exactly like a shared borrow --
hence the borrowed value can still be read.

As discussed earlier, this RFC itself only introduces these two-phase
borrows in a limited way. Specifically, we extend the MIR with a new
kind of borrow (written `mut2`, for two-phase), and we generate those
new kinds of borrows when lowering method calls.

To understand how two-phased borrows help, let's revisit our two
examples. We'll start with the motivating example,
`vec.push(vec.len())`. When this expression is desugared, the
resulting reference is stored into a temporary, `tmp0`.  Therefore,
until `tmp0` is referenced again, `vec` is only considered
**reserved**:

```rust
/* 0 */ tmp0 = &mut2 vec;       // reservation of `vec` starts here
/* 1 */ tmp1 = &vec;
/* 2 */ tmp2 = Vec::len(tmp1);
/* 3 */ Vec::push(tmp0, tmp2); // first use of `tmp0`, upgrade is here
```

The first use of `tmp0` is on line 3, and hence the mutable borrow
begins then, and lasts until the end of the borrow region. Crucially,
lines 1 and 2 (which did a shared borrow of `vec`) took place during
the reservation period, and hence no error results. This is because a
reservation is equivalent to a shared borrow, and multiple shared
borrows are allowed.

Next, let's consider the sneaky example, where the argument attempts
to mutate the vector that is being used in the receiver:

```rust
let mut v: Vec<String> = vec![format!("Hello, ")];
let s: String = format!("foo");
v[0].push_str({ v.push(s); "World!" });
//              ^^^^^^^^^ sneaky attempt to mutate `v`
```

In this case, if we examine the resulting MIR, we can see that the
borrow of `v` is almost **immediately** used, as part of the
`IndexMut` operation:

```rust
// First evaluate `v[0]` to get a `&mut String`:
tmp0 = &mut2 v;
tmp1 = IndexMut::index_mut(tmp0, 0); // tmp0 used here!
tmp2 = tmp1;

// Next, evaluate `{ v.push(s); "World!" }` block:
tmp3 = &mut2 v; // <-- Error! mutable borrow of `v` is active.
... // see above
```

This implies that the mutable borrow will be active later on, when `v`
is borrowed again during the arguments, and hence an error is still
reported.

Note that this same treatment will also rule out some "harmless"
examples, such as this one:

```rust
v[0].push_str(&format!("{}", v.len()));
```

This might seem analogous to example 1, but in this case the mutable
borrow of `v` is "activated" by the indexing, and hence `v` is
considered mutably borrowed when `v.len()` is called, not reserved,
which results in an error.

# Detailed design
[design]: #detailed-design

### New MIR form for two-phase borrows

Currently, the MIR rvalue for borrows has one of three forms (these
are internal syntax only, naturally, since MIR doesn't have a defined
written representation)

    &'a <lvalue>
    &'a mut <lvalue>
    &'a unique <lvalue>
    
In either case, the rvalue returns a reference with lvalue `'a` that
refers to the address of `lvalue` (an `lvalue` is a path that leads to
memory). This can be either a shared, mutable, or unique reference
(unique references are an internal concept that appears only in MIR;
they are used when desugaring closures, but there is no direct
equivalent in Rust surface syntax).

This RFC proposes adding a third form: `&'a mut2 <lvalue>`. Like
`&unique` borrows, this would be used by the compiler when desugaring
and would not have a direct user representation for the time
being. For most purposes, an `&mut2` borrow would act precisely the
same as an `&mut` borrow; the borrow checker however would treat it
differently, as described below.

### When are two-phase borrows used

Two-phase borrows would be used in the specific case of desugaring a
call to an `&mut self` method. Currently, in the initially generated
MIR, calls to such methods *always* have a "auto-mut-ref" inserted
(this is because `vec.push()`, where `vec: &mut Vec<i32>`, is
considered a *borrow* of `vec`, not a move). This "auto-mut-ref" will
be changed from an `&mut` to an `&mut2`.

### Integrating reserved borrows into the borrow checker

#### Existing MIR borrowck algorithm

The proposed fix for this problem is described in terms of a MIR-based
borrowck (which is coming soon). The basic structure of the existing
borrow checker, transposed onto MIR, is as follows:

- Every borrow in MIR always has the same form:
  - `lv1 = &'r lv2` or `lv1 = &'r mut lv2`, where:
    - `lv1` and `lv2` are MIR lvalues (path naming a memory location)
    - `'r` is the duration of the borrow
- Let each borrow be named by its position `P`, which has the form
  `BB/n`, where `BB` is the basic block containing the borrow
  statement and `n` is the index within that basic block.
- The borrow at position `P` is then considered **live** for all points
  reachable from `P` without passing through the end of the region
  `'r`.
  - The full set of borrows live at a given point can be readily
    computed using a standard data-flow analysis.
- For each **write** to an lvalue `lv_w` at point `P`:
  - A **write** is either a mutable borrow `&mut lv_w` or an assignment `lv_w = ...`
  - It is an error if there is any borrow (mutable or shared) of some path `lv_b` that is **live** at `P`
    where `lv_b` may overlap `lv_w`
- For each **read** from an lvalue `lv_r` at point `P`:
  - A **read** is any use of `lv_r` as an operand.
  - It is an error if there is any mutable borrow of some path `lv_b` that is **live** at `P`
    where `lv_b` may overlap `lv_r`

#### Proposed change

When the borrow checker encounters a `mut2` borrow, it will handle it
in a slightly different way. Because of the limited places where `mut2` borrows
are generated, we know that they will only ever be encountered in a statement
that assigns them to a MIR temporary:

```
tmp = &'r mut2 lv
```

In that case, the path `lv` would initially be considered
**reserved**. The temporary `tmp` will only be used once, as an
argument to the actual call: at that point, the path `lv` will be
considered **mutably borrowed**.

In terms of the safety checks, reservations act just as a shared
borrow does. Therefore, a write to `lv` at point `P` is illegal if
there is any active borrow **or** in-scope reservation of `lv` at the
point `P`. Similarly, a read from `lv` at point `P` is legal if there
exists a reservation (but not with a mutable borrow).

There is one new check required. At the point `Q` where a mutable
borrow is activated, we must check that there are no active borrows or
reservations in scope (other than the reservation being upgraded). Otherwise,
a test such as this might pass:

```rust
fn foo<'a>(x: &'a Vec<i32>) -> &'a i32 { &x[0] }

let mut v = vec![0, 1, 2];
let p;
v.push({p = foo(&v); 3});
use(*p);
```

When desugared into MIR, this would look something like:

```
tmp0 = &'a mut2 v;   // reservation begins
tmp1 = &'b v;       // shared borrow begins; allowed, because `v` is reserved
p = foo(tmp1);
Vec::push(tmp0, 3); // mutable borrow activated
EndRegion('a);      // mutable borrow ends
tmp2 = *p;          // shared borrow still valid!
use(tmp2) 
EndRegion('b);
```

Note that, here, we created a borrow of `v[0]` *before* we called
`Vec::push()`, and we continue to use it afterwards. This should not
be accepted, but it could be without this additional check at the
activation point. In particular, at the time that the shared borrow
*starts*, `v` is reserved; the mutable borrow of `v` is activated
later, but still within the scope of the shared borrow. (In today's
borrow checker, this cannot happen, so we only check at the start of a
borrow whether other borrows are in scope.)

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

For the most part, because this change is so targeted, it seems that
discussion of how it works is out of scope for introductory texts such
as The Rust Programming Language or Rust By Example. In particular,
the idea simply makes code that seems intuitively like it *should*
work (e.g., `vec.push(vec.len())`) work.

However, there are a few related topics which likely *might* make sense
to cover at some point in works like this:

- People will likely first encounter surprises when they attempt more
  complicated method calls that are not covered by this proposal, such
  as the `v[0].push_str(&format!("{}", v.len()));` example. In that
  case, a simple desugaring can be used to show why the compiler
  rejects this code -- in particular, a comparison with the errorneous
  examples may be helpful. A keen observer may note the contrast with
  `vec.push(vec.len())`, but such an observer can be referred to the
  reference. =)
- One interesting point that came up in discussing this example is
  that many people expect that `vec.push(vec.len())` would be
  desugared as follows:

  ```
  let tmp = vec.len();
  vec.push(tmp)
  ```

  In particular, note that `vec`, in this desugaring, is not assigned
  to a temporary.  This is in fact not how the language works (as
  discussed in more detail under the Alternatives section); instead,
  `vec` is treated like any other argument. It is evaluated to a
  temporary, and autorefs etc are applied. It may be worth covering
  this sort of example when doing an in-depth explanation of how
  method desugaring works.

Coverage of these rules seems most appropriate for the Rust reference,
as part of detailed general coverage on how MIR desugaring and the
borrow checker work. At the moment, no such coverage exists, but this
would be a logical part of it. In that context, explaining it in a
similar fashion to how the RFC presents the change seems appropriate.

# Drawbacks
[drawbacks]: #drawbacks

The obvious downside of this proposal is that it is narrowly targeted
at the method call form. This means that "manual desugarings" of
method calls will not necessarily work, particularly if the user
faithfully follows what the compiler does. There are a number of
reasons to think this will be not be a very big deal in practice:

- There is rarely a desire to do manual desugaring of method calls anyway.
- In practice, when a desugaring *is* needed, people have a lot of
  latitude to adjust the ordering of statements and so forth, and
  hence they can achieve the effect that they need (in fact, every
  time that you are forced to rewrite an instance of the
  `vec.push(vec.len())` pattern to save `vec.len()` into a temporary,
  you are doing a partial desugaring of this kind).
- **Truly** faithful desugarings are rare in any case. As discussed in
  the How We Teach This section, many people overlook the role of
  autoref and the precise evaluation order. Fewer still will get the
  precise lifetime of temporaries correctly or other details. This is
  not a big deal.

Nonetheless, this change slightly widens the gap between the surface
language and the underlying "desugared" view that MIR takes, and in
general that is to be avoided. The Alternatives section discuses some
possible future extensions that could be used to remove that gap.

# Alternatives
[alternatives]: #alternatives

As discussed earlier, a number of major alternative designs have been
put forward to address nested method calls. This proposal is intended
to be forwards compatible with all of them, but to adopt none of them
in particular. We cover now each alternative and explain why we did
not want to adopt it in this RFC.

### Modifying the desugaring to evaluate receiver after arguments 

One option is to modify the desugaring for method calls. Currently,
a call like `a.foo(b..z)` is always desugared into something like:

- process `a` and apply any autoref etc, resulting in `tmp0`
- evaluate `b..z` to a temporary, resulting in `tmp1..tmpN`
- invoke `foo(tmp0..tmpN)`

However, we could say that, under some set of circumstances,
we will evaluate `a` later:

- evaluate `b..z` to a temporary, resulting in `tmp1..tmpN`
- process `a` and apply any autoref etc, resulting in `tmp0`
- invoke `foo(tmp0..tmpN)`

Due to backwards compatibility constraints, there are some limits to
how often we could do this reordering. For example, we clearly cannot
change the desugaring of complex, side-effecting expressions like
`a().foo(b())`. In fact, even simple expressions like `a.foo(b)` might
be a breaking change, if the method is declared as `fn(self)`
([play link](https://is.gd/yz3zFq)):

```rust
trait Foo {
  fn foo(self, a: ()) -> Self;
}

impl Foo for i32 {
  fn foo(self, a: ()) -> Self {
    self
  }
}

let mut a = 3;
let b = a.foo({ a += 1; () }); // returns 3
```

In effect, the goal would be to come up with some rules that limit the
cases under consideration to cases that would currently result in an
error. One proposed set of rules might be:

- the invoked method `foo()` is an `&mut self` method
- the receiver is simply a reference to a local variable `a`

This would cause, for example, `vec.push(vec.len())` to use the new
ordering, and hence to be accepted. However, `v[0].push(...)` would
not use the new ordering.

This option strikes many as being simpler than the one proposed here.
It is perhaps simpler to explain, especially, since it doesn't
introduce any new concepts -- the borrow checker works as it ever did,
and we already have to do desugaring *somehow*, we're just doing it
differently in this case. And in particular we're only affecting cases
where autoref -- a non-trivial desugaring -- applies.

However, this option can also result in some surprises of its own.
For example, consider a twist on the previous example, where
the method `foo` is declared as `&mut self` instead:

```rust
trait Foo {
  fn foo(&mut self, a: ()) -> Self;
}

impl Foo for i32 {
  fn foo(&mut self, a: ()) -> Self {
    *self
  }
}

let mut a = &mut 3;
let b = a.foo({ a = &mut 4; () }); // returns 4
```

Currently, this code will not compile. Under the proposal, however, it
would compile, because (1) the method is `&mut self` and (2) the
receiver is a simple variable reference `a`. Interestingly, now that
we changed the method to `&mut self`, we can suddenly see the
side-effects of evaluating the argument.

On balance, it seems better to this author to have the borrow checker
analysis be more complex than the desugaring and execution order.

### Permit more things during the "restricted" period

The current notion of a 'restricted' borrow is identical to a shared
borrow. However, we could in principle permit *more* things during the
restricted period -- basically we could permit anything that does not
invalidate the reference we created. In that case, we might fruitfully
enable two-phased borrows for shared references as well. In practice,
this means that we could permit writes to the borrowed content (which
are forbidden by this proposal). An example of code that would work as
a result is the following:

```rust
// pretend you could define an inherent method on integers
// for a second, just to keep code snippet simple
impl i32 {
    fn increment(&mut self, v: i32) -> i32 {
        *self += v;
        *self // returns new value
    }
}
                                            
fn foo() {
    let mut x = 0;
    let y = x.increment(x.increment(1)); // what result do you expect from this?
    println!("{}", y);
}
```

The call to `x.increment(x.increment(1))` would thus desugar to the following MIR:

```
tmp0 = &mut2 x;
tmp1 = &mut2 x;
tmp2 = 1;
tmp3 = i32::increment(tmp1, tmp2); // activates tmp1
i32::increment(tmp0, tmp3); // activates tmp0
```

Under the existing proposal, this is illegal, because `x` is
considered "reserved" when `tmp1` is created, and an `&mut2` borrow is
not permitted when the lvalue being borrowed has been reserved. If we
made restrictions more permissive, we might accept this code; it would
output `2`.

We opted against this variation for several reasons:

- It makes the borrow checker more complex by introducing not only
  two-phase borrows, but a new set of restrictions that must be worked
  out in detail. The current RFC leverages the existing category of
  shared borrows.
- The main gain here is the ability to intersperse two mutable calls
  (as in the example), or to have an outer shared borrow with an inner
  mutable borrow. In general, this implies that there is some careful
  ordering of mutation going on here: in particular, the outer method
  call will observe the state changes made by the inner calls. This
  feels like a case where it is *helpful* to have the user pull the
  two calls apart, so that their relative side-effects are clearly
  visible.
  
Of course, it would be possible to loosen the rules in the future.
  
### A broader user of two-phase borrows

The initial proposal for two-phased borrows (made in
[this blog post][]) was more expansive. In particular, it aimed to
convert **all mutable borrows** into two-phase borrows at the MIR
level.  Given the way that MIR is generated, this meant that users
would be able to observe these two phases in some cases. For example,
the following code would have type-checked, whereas it would not today
or under this RFC:

[a blog post]: http://smallcultfollowing.com/babysteps/blog/2017/03/01/nested-method-calls-via-two-phase-borrowing/

```rust
let tmp0 = &mut vec;   // `vec` is reserved
let tmp1 = vec.len();  // shared borrow of vec; ok
Vec::push(tmp0, tmp1); // mutable borrow of `vec` is activated
```

The aim here was specifically to support the desugared form of a
method call.

The current RFC backs down from this more aggressive posture. Treating
all mutable borrows as potentially deferred would make them something
that everyday users would encounter, and we didn't feel satisfied with
the "mental model" that resulted. In particular, because of how MIR is
generated, deferred borrows would be almost immediately activated in
most scenarios.  They would only work when a borrow was *immediately*
assigned into a variable as part of a `let` declaration. This means,
for example, that these two bits of code would have been treated
differently:

```rust
let x = &mut vec; // reserved

// versus:

let x;
x = &mut vec; // immediately activated
```

The reason for this distinction cannot be explained except by examining the desugarings
into MIR; if you do so, you will see that the second case introduces an intermediate temporary:

```
tmp0 = &mut vec; // reservation starts
x = tmp0; // borrow is activated
```

The root of the problem is that the current RFC is proposing an
analysis that is not done on **types** but rather on MIR variables and
points in the control-flow graph. This means that (for example)
whether a borrow is activated is affected by "no-ops" like `let x = y`
(which would be considered a use of `y`).

Therefore, introducing two-phased borrows **outside** of method-call
desugaring form doesn't feel like the right approach. (But, if they
are limited to method-call desugaring, as ths RFC proposes, then they
are a simple and effective mechanism without broader impact.)

### Borrowing for the future

One of the initial proposals for how to think about nested method
calls was in terms of "borrowing for the future". Currently, whenever
you have a borrow, the resulting reference is "immediately
usable". That is, the lifetime of the reference must include the point
of the borrow. Borrowing for the future proposes to loosen that rule,
allowing a borrow to result in a reference that can't be *immediately*
used, but can only be used at some future point. In the meantime, the
path that was borrowed must be considered to be *reserved* (in roughly
the same sense as this RFC uses it), in order to ensure that the
reference is not invalidated.

To see how this might work, consider the naively desugared version of
`vec.push(vec.len())`, but with explicit labels for the lifetime of
every little part (and also for the lifetime of a borrow):
 
 ```rust
'call: {
  let v: &'invoke mut Vec<usize>;
  let l: usize;
  'eval_args: {
    'eval_v: { v = &'eval_l vec; }
    'eval_l: { l = Vec::len(v); }
  }
  'invoke: { Vec::push(v, l); }
}
```

Here you can see that the borrow `v = &'invoke mut vec` is borrowing `vec`
for a lifetime (`'invoke`) that has not yet started -- but which will start
in the future. This is basically saying, "make a reference that we will give
to this function, but we won't use in the meantime".

Since the reference `v` is not in active use yet, we can use looser
restrictions.  We still need to consider the path `vec` to be
"reserved", so that `v` doesn't get evaluated. The idea is that we are
evaluating the path to a pointer right then and there, so we need to
be sure that this pointer remains valid. We wouldn't want people to
send `vec` to another thread or something.
                   
It seems plausible that these rules could be integrated into the
notion of non-lexical lifetimes. At present, the
[non-lexical lifetimes proposal][nll] still includes the rule that
borrows must be immediately active (in particular, at each point P
where a variable is live, all of the regions in its type must include
P). But this could be changed to a rule that says that the regions
must either include P or be a future region of the kind shown here.
Clearly, the details will need to be worked out, but this would then
present a more cohesive model that we could teach to users (in short,
when you make a reference, the span of the code where the reference is
in active use is restricted, and the code leading up to that span
treats the value as having been shared).
                   
[nll]: http://smallcultfollowing.com/babysteps/blog/2017/02/21/non-lexical-lifetimes-using-liveness-and-location/

### Ref2

In the internals thread, arielb1 had [an interesting proposal][ref2]
that they called "two-phase lifetimes". The goal was precisely to take
the "two-phase" concept but incorporate it into lifetime inference,
rather than handling it in borrow checking as I present here. The idea
was to define a type `RefMut<'r, 'w, T>` (original `Ref2Φ<'immut,
'mutbl, T>`) which stands in for a kind of "richer" `&mut` type
(originally, `&T` was unified as well, but that introduces
complications because `&T` types are `Copy`, so I'm leaving that
out). In particular, `RefMut` has two lifetimes, not just one:

- `'r` is the "read" lifetime. It includes every point where the reference
   may later be used.
- `'w` is a subset of `'r` (that is, `'r: 'w`) which indicates the "write" lifetime.
  This includes those points where the reference is actively being written.
 
We can then conservatively translate a `&'a mut T` type into
`RefMut<'a, 'a, T>` -- that is, we can use `'a` for both of the two
lifetimes. This is what we would do for any `&mut` type that appears
in a struct declaration or fn interface. But for `&mut T` types within
a fn body, we can infer the two lifetimes somewhat separately: the
`'r` lifetime is computed just as I described in my
[NLL post][NLL]. But the `'w` lifetime only needs to include those
points where a write occurs. The borrow check would then guarantee
that the `'w` regions of every `&mut` borrow is disjoint from the `'r`
regions of every other borrow (and from shared borrows).

This proposal has a lot of potential applications, but each of them
introduces some complications, and would require singificant further
thought. Let's cover them in more detail.

#### Discontinuous borrows

This proposal accepts more programs than the one I outlined. In
particular, it accepts the example with interleaved reads and writes
that we saw earlier. Let me give that example again, but annotation
the regions more explicitly:

```rust
/* 0 */ let mut i = 0;
/* 1 */ let p: RefMut<{2-5}, {3,5}, i32> = &mut i;
//                    ^^^^^  ^^^^^
//                     'r     'w
/* 2 */ let j = i;  // just in 'r
/* 3 */ *p += 1;    // must be in 'w
/* 4 */ let k = i;  // just in 'r
/* 5 */ *p += 1;    // must be in 'w
```

As you can see here, we would infer the write region to be just the
two points 3 and 5. This is precisely those portions of the CFG where
writes are happening -- and not the gaps in between, where reads are
permitted.

As you might have surmised, these sorts of "discontinuous" borrows
represent a kind of "step up" in the complexity of the system. If it
were vital to accept examples with interleaved writes like the
previous one, then this wouldn't bother me (NLL also represents such a
step, for example, but it seems clearly worth it). But given that the
example is artificial and not a pattern I have ever seen arise in
"real life", it seems like we should try to avoid growing the
underlying complexity of the system if we can.

To see what I mean about a "step up" in complexity, consider how we
would integrate this proposal into lifetime inference. The current
rules treat all regions equally, but this proposal seems to imply that
regions have "roles".  For example, the `'r` region captures the
"liveness" constraints that I described in the original NLL
proposal. Meanwhile the `'w` region captures "activity".

(Since we would always convert a `&'a mut T` type into `RefMut<'a, 'a,
T>`, all regions in struct parameters would adopt the more
conservative "liveness" role to start. This is good because we
wouldn't want to start allowing "holes" in the lifetimes that unsafe
code is relying on to prevent access from the outside. It would
however be possible for type inference to use a `RefMut<'r, 'w ,T>`
type as the value for a type parameter; I don't yet see a way for that
to cause any surprises, but perhaps it can if you consider
specialization and other non-parametric features.)

Another example of where this "complexity step" surfaces came from
[Ralf Jung][rjung]. As you may know, Ralf is working on a
formalization of Rust as part of the [RustBelt project][rb] (if you're
interested, there is video available of a
[great introduction to this work][am] which Ralf gave at the Rust
Paris meetup). In any case, their model is a kind of generalization of
Rust, in that it can accept a lot of programs that standard Rust
cannot (it is intended to be used for assigning types to unsafe code
as well as safe code). The two-phase borrow proposal that I describe
here should be able to fit into that system in a fairly
straightforward way. But if we adopted discontinuous regions, that
would require making Ralf's system more expressive. This is not
necessarily an argument against doing it, but it does show that it
makes the Rust system qualitatively more complex to reason about.

[rb]: http://plv.mpi-sws.org/rustbelt/
[rjung]: https://www.ralfj.de/blog/
[am]: https://air.mozilla.org/rust-paris-meetup-35-2017-01-19/

If all this talk of "steps in complexity" seems abstract, I think that
the most immediate way it will surface is when we try to
**teach**. Supporting discontinous borrows just makes it that much
harder to craft small examples that show how borrowing works. It will
make the system feel more mysterious, since the underlying rules are
indeed more complex and thus harder to "intuit" on your own. Getting
these details right is a significant design challenge outside the
scope of this RFC.

#### Downgrading mutable to shared

Another goal of the proposal was to (perhaps someday) support the
"downgrade-mut-to-shared" pattern, in which a function takes in a
mutable reference but returns a shared reference:

```rust
fn get_something(&mut self) -> &T {
    self.data = ...;
    &self.data
}    
```

In the case of this function, we do indeed require a mutable borrow of
`self` to start -- since we update `self.data` -- but once
`get_something()` returns, a simple shared borrow would suffice (as is
the case for the pseudo-code above). It is conceivable that such a
scenario could be handled by giving `&mut self` a "write" lifetime
that is confined to the call itself, but a bigger "read" lifetime.

However, there are other cases (that exist in active use today) of
functions that take an `&mut self` and return an `&T` where it would
*not* be safe to treat `self` as shared after the function
returns. For example, one could easily wrap the existing
`Mutex::get_mut` function to have a signature like this; `get_mut()`
works by taking an `&mut` reference and giving access to the interior
of the mutex **without locking it**. This is only possible because
`get_mut()` can assume that `self` will remain **mutably** borrowed
until you are done using that data.  See
[this post on the internals thread](https://internals.rust-lang.org/t/blog-post-nested-method-calls-via-two-phase-borrowing/4886/33?u=nikomatsakis)
for more details.

Therefore, it seems that some form of user annotation would be
required to enable this pattern. This implies that the two lifetimes
of the `Ref2` type would have to be exposed to end-users, or other
annotations are needed. Just as with discontinuous borrows, designing
such a system is a significant design challenge outside the scope of
this RFC.

# Unresolved questions
[unresolved]: #unresolved-questions

None as yet..
R
