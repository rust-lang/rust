# Async closures/"coroutine-closures"

Please read [RFC 3668](https://rust-lang.github.io/rfcs/3668-async-closures.html) to understand the general motivation of the feature. This is a very technical and somewhat "vertical" chapter; ideally we'd split this and sprinkle it across all the relevant chapters, but for the purposes of understanding async closures *holistically*, I've put this together all here in one chapter.

## Coroutine-closures -- a technical deep dive

Coroutine-closures are a generalization of async closures, being special syntax for closure expressions which return a coroutine, notably one that is allowed to capture from the closure's upvars.

For now, the only usable kind of coroutine-closure is the async closure, and supporting async closures is the extent of this PR. We may eventually support `gen || {}`, etc., and most of the problems and curiosities described in this document apply to all coroutine-closures in general.

As a consequence of the code being somewhat general, this document may flip between calling them "async closures" and "coroutine-closures". The future that is returned by the async closure will generally be called the "coroutine" or the "child coroutine".

### HIR

Async closures (and in the future, other coroutine flavors such as `gen`) are represented in HIR as a `hir::Closure`.
The closure-kind of the `hir::Closure` is `ClosureKind::CoroutineClosure(_)`[^k1], which wraps an async block, which is also represented in HIR as a `hir::Closure`.
The closure-kind of the async block is `ClosureKind::Closure(CoroutineKind::Desugared(_, CoroutineSource::Closure))`[^k2].

[^k1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_ast_lowering/src/expr.rs#L1147>

[^k2]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_ast_lowering/src/expr.rs#L1117>

Like `async fn`, when lowering an async closure's body, we need to unconditionally move all of the closures arguments into the body so they are captured. This is handled by `lower_coroutine_body_with_moved_arguments`[^l1]. The only notable quirk with this function is that the async block we end up generating as a capture kind of `CaptureBy::ByRef`[^l2]. We later force all of the *closure args* to be captured by-value[^l3], but we don't want the *whole* async block to act as if it were an `async move`, since that would defeat the purpose of the self-borrowing of an async closure.

[^l1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_ast_lowering/src/item.rs#L1096-L1100>

[^l2]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_ast_lowering/src/item.rs#L1276-L1279>

[^l3]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_hir_typeck/src/upvar.rs#L250-L256>

### `rustc_middle::ty` Representation

For the purposes of keeping the implementation mostly future-compatible (i.e. with gen `|| {}` and `async gen || {}`), most of this section calls async closures "coroutine-closures".

The main thing that this PR introduces is a new `TyKind` called `CoroutineClosure`[^t1] and corresponding variants on other relevant enums in typeck and borrowck (`UpvarArgs`, `DefiningTy`, `AggregateKind`).

[^t1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_type_ir/src/ty_kind.rs#L163-L168>

We introduce a new `TyKind` instead of generalizing the existing `TyKind::Closure` due to major representational differences in the type. The major differences between `CoroutineClosure`s can be explored by first inspecting the `CoroutineClosureArgsParts`, which is the "unpacked" representation of the coroutine-closure's generics.

#### Similarities to closures

Like a closure, we have `parent_args`, a `closure_kind_ty`, and a `tupled_upvars_ty`. These represent the same thing as their closure counterparts; namely: the generics inherited from the body that the closure is defined in, the maximum "calling capability" of the closure (i.e. must it be consumed to be called, like `FnOnce`, or can it be called by-ref), and the captured upvars of the closure itself.

#### The signature

A traditional closure has a `fn_sig_as_fn_ptr_ty` which it uses to represent the signature of the closure. In contrast, we store the signature of a coroutine closure in a somewhat "exploded" way, since coroutine-closures have *two* signatures depending on what `AsyncFn*` trait you call it with (see below sections).

Conceptually, the coroutine-closure may be thought as containing several different signature types depending on whether it is being called by-ref or by-move.

To conveniently recreate both of these signatures, the `signature_parts_ty` stores all of the relevant parts of the coroutine returned by this coroutine-closure. This signature parts type will have the general shape of `fn(tupled_inputs, resume_ty) -> (return_ty, yield_ty)`, where `resume_ty`, `return_ty`, and `yield_ty` are the respective types for the *coroutine* returned by the coroutine-closure[^c1].

[^c1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_type_ir/src/ty_kind/closure.rs#L221-L229>

The compiler mainly deals with the `CoroutineClosureSignature` type[^c2], which is created by extracting the relevant types out of the `fn()` ptr type described above, and which exposes methods that can be used to construct the *coroutine* that the coroutine-closure ultimately returns.

[^c2]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_type_ir/src/ty_kind/closure.rs#L362>

#### The data we need to carry along to construct a `Coroutine` return type

Along with the data stored in the signature, to construct a `TyKind::Coroutine` to return, we also need to store the "witness" of the coroutine.

So what about the upvars of the `Coroutine` that is returned? Well, for `AsyncFnOnce` (i.e. call-by-move), this is simply the same upvars that the coroutine returns. But for `AsyncFnMut`/`AsyncFn`, the coroutine that is returned from the coroutine-closure borrows data from the coroutine-closure with a given "environment" lifetime[^c3]. This corresponds to the `&self` lifetime[^c4] on the `AsyncFnMut`/`AsyncFn` call signature, and the GAT lifetime of the `ByRef`[^c5].

[^c3]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_type_ir/src/ty_kind/closure.rs#L447-L455>

[^c4]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/library/core/src/ops/async_function.rs#L36>

[^c5]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/library/core/src/ops/async_function.rs#L30>

#### Actually getting the coroutine return type(s)

To most easily construct the `Coroutine` that a coroutine-closure returns, you can use the `to_coroutine_given_kind_and_upvars`[^helper] helper on `CoroutineClosureSignature`, which can be acquired from the `CoroutineClosureArgs`.

[^helper]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_type_ir/src/ty_kind/closure.rs#L419>

Most of the args to that function will be components that you can get out of the `CoroutineArgs`, except for the `goal_kind: ClosureKind` which controls which flavor of coroutine to return based off of the `ClosureKind` passed in -- i.e. it will prepare the by-ref coroutine if `ClosureKind::Fn | ClosureKind::FnMut`, and the by-move coroutine if `ClosureKind::FnOnce`.

### Trait Hierarchy

We introduce a parallel hierarchy of `Fn*` traits that are implemented for . The motivation for the introduction was covered in a blog post: [Async Closures](https://hackmd.io/@compiler-errors/async-closures).

All currently-stable callable types (i.e., closures, function items, function pointers, and `dyn Fn*` trait objects) automatically implement `AsyncFn*() -> T` if they implement `Fn*() -> Fut` for some output type `Fut`, and `Fut` implements `Future<Output = T>`[^tr1].

[^tr1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_next_trait_solver/src/solve/assembly/structural_traits.rs#L404-L409>

Async closures implement `AsyncFn*` as their bodies permit; i.e. if they end up using upvars in a way that is compatible (i.e. if they consume or mutate their upvars, it may affect whether they implement `AsyncFn` and `AsyncFnMut`...)

#### Lending

We may in the future move `AsyncFn*` onto a more general set of `LendingFn*` traits; however, there are some concrete technical implementation details that limit our ability to use `LendingFn` ergonomically in the compiler today. These have to do with:

- Closure signature inference.
- Limitations around higher-ranked trait bounds.
- Shortcomings with error messages.

These limitations, plus the fact that the underlying trait should have no effect on the user experience of async closures and async `Fn` trait bounds, leads us to `AsyncFn*` for now. To ensure we can eventually move to these more general traits, the precise `AsyncFn*` trait definitions (including the associated types) are left as an implementation detail.

#### When do async closures implement the regular `Fn*` traits?

We mention above that "regular" callable types can implement `AsyncFn*`, but the reverse question exists of "can async closures implement `Fn*` too"? The short answer is "when it's valid", i.e. when the coroutine that would have been returned from `AsyncFn`/`AsyncFnMut` does not actually have any upvars that are "lent" from the parent coroutine-closure.

See the "follow-up: when do..." section below for an elaborated answer. The full answer describes a pretty interesting and hopefully thorough heuristic that is used to ensure that most async closures "just work".

### Tale of two bodies...

When async closures are called with `AsyncFn`/`AsyncFnMut`, they return a coroutine that borrows from the closure. However, when they are called via `AsyncFnOnce`, we consume that closure, and cannot return a coroutine that borrows from data that is now dropped.

To work around this limitation, we synthesize a separate by-move MIR body for calling `AsyncFnOnce::call_once` on a coroutine-closure that can be called by-ref.

This body operates identically to the "normal" coroutine returned from calling the coroutine-closure, except for the fact that it has a different set of upvars, since we must *move* the captures from the parent coroutine-closure into the child coroutine.

#### Synthesizing the by-move body

When we want to access the by-move body of the coroutine returned by a coroutine-closure, we can do so via the `coroutine_by_move_body_def_id`[^b1] query.

[^b1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_mir_transform/src/coroutine/by_move_body.rs#L1-L70>

This query synthesizes a new MIR body by copying the MIR body of the coroutine and inserting additional derefs and field projections[^b2] to preserve the semantics of the body.

[^b2]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_mir_transform/src/coroutine/by_move_body.rs#L131-L195>

Since we've synthesized a new def id, this query is also responsible for feeding a ton of other relevant queries for the MIR body. This query is `ensure()`d[^b3] during the `mir_promoted` query, since it operates on the *built* mir of the coroutine.

[^b3]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_mir_transform/src/lib.rs#L339-L342>

### Closure signature inference

The closure signature inference algorithm for async closures is a bit more complicated than the inference algorithm for "traditional" closures. Like closures, we iterate through all of the clauses that may be relevant (for the expectation type passed in)[^deduce1].

To extract a signature, we consider two situations:
* Projection predicates with `AsyncFnOnce::Output`, which we will use to extract the inputs and output type for the closure. This corresponds to the situation that there was a `F: AsyncFn*() -> T` bound[^deduce2].
* Projection predicates with `FnOnce::Output`, which we will use to extract the inputs. For the output, we also try to deduce an output by looking for relevant `Future::Output` projection predicates. This corresponds to the situation that there was an `F: Fn*() -> T, T: Future<Output = U>` bound.[^deduce3]
    * If there is no `Future` bound, we simply use a fresh infer var for the output. This corresponds to the case where one can pass an async closure to a combinator function like `Option::map`.[^deduce4]

[^deduce1]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_hir_typeck/src/closure.rs#L345-L362>

[^deduce2]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_hir_typeck/src/closure.rs#L486-L487>

[^deduce3]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_hir_typeck/src/closure.rs#L517-L534>

[^deduce4]: <https://github.com/rust-lang/rust/blob/5ca0e9fa9b2f92b463a0a2b0b34315e09c0b7236/compiler/rustc_hir_typeck/src/closure.rs#L575-L590>

We support the latter case simply to make it easier for users to simply drop-in `async || {}` syntax, even when they're calling an API that was designed before first-class `AsyncFn*` traits were available.

#### Calling a closure before its kind has been inferred

We defer[^call1] the computation of a coroutine-closure's "kind" (i.e. its maximum calling mode: `AsyncFnOnce`/`AsyncFnMut`/`AsyncFn`) until the end of typeck. However, since we want to be able to call that coroutine-closure before the end of typeck, we need to come up with the return type of the coroutine-closure before that.

[^call1]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_hir_typeck/src/callee.rs#L169-L210>

Unlike regular closures, whose return type does not change depending on what `Fn*` trait we call it with, coroutine-closures *do* end up returning different coroutine types depending on the flavor of `AsyncFn*` trait used to call it. 

Specifically, while the def-id of the returned coroutine does not change, the upvars[^call2] (which are either borrowed or moved from the parent coroutine-closure) and the coroutine-kind[^call3] are dependent on the calling mode.

[^call2]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_type_ir/src/ty_kind/closure.rs#L574-L576>

[^call3]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_type_ir/src/ty_kind/closure.rs#L554-L563>

We introduce a `AsyncFnKindHelper` trait which allows us to defer the question of "does this coroutine-closure support this calling mode"[^helper1] via a trait goal, and "what are the tupled upvars of this calling mode"[^helper2] via an associated type, which can be computed by appending the input types of the coroutine-closure to either the upvars or the "by ref" upvars computed during upvar analysis.

[^helper1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/library/core/src/ops/async_function.rs#L135-L144>

[^helper2]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/library/core/src/ops/async_function.rs#L146-L154>

#### Ok, so why?

This seems a bit roundabout and complex, and I admit that it is. But let's think of the "do nothing" alternative -- we could instead mark all `AsyncFn*` goals as ambiguous until upvar analysis, at which point we would know exactly what to put into the upvars of the coroutine we return. However, this is actually *very* detrimental to inference in the program, since it means that programs like this would not be valid:

```rust!
let c = async || -> String { .. };
let s = c().await;
// ^^^ If we can't project `<{c} as AsyncFn>::call()` to a coroutine, then the `IntoFuture::into_future` call inside of the `.await` stalls, and the type of `s` is left unconstrained as an infer var.
s.as_bytes();
// ^^^ That means we can't call any methods on the awaited return of a coroutine-closure, like... at all!
```

So *instead*, we use this alias (in this case, a projection: `AsyncFnKindHelper::Upvars<'env, ...>`) to delay the computation of the *tupled upvars* and give us something to put in its place, while still allowing us to return a `TyKind::Coroutine` (which is a rigid type) and we may successfully confirm the built-in traits we need (in our case, `Future`), since the `Future` implementation doesn't depend on the upvars at all.

### Upvar analysis

By and large, the upvar analysis for coroutine-closures and their child coroutines proceeds like normal upvar analysis. However, there are several interesting bits that happen to account for async closures' special natures:

#### Forcing all inputs to be captured

Like async fn, all input arguments are captured. We explicitly force[^f1] all of these inputs to be captured by move so that the future coroutine returned by async closures does not depend on whether the input is *used* by the body or not, which would impart an interesting semver hazard.

[^f1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_hir_typeck/src/upvar.rs#L250-L259>

#### Computing the by-ref captures

For a coroutine-closure that supports `AsyncFn`/`AsyncFnMut`, we must also compute the relationship between the captures of the coroutine-closure and its child coroutine. Specifically, the coroutine-closure may `move` a upvar into its captures, but the coroutine may only borrow that upvar.

We compute the "`coroutine_captures_by_ref_ty`" by looking at all of the child coroutine's captures and comparing them to the corresponding capture of the parent coroutine-closure[^br1]. This `coroutine_captures_by_ref_ty` ends up being represented as a `for<'env> fn() -> captures...` type, with the additional binder lifetime representing the "`&self`" lifetime of calling `AsyncFn::async_call` or `AsyncFnMut::async_call_mut`. We instantiate that binder later when actually calling the methods.

[^br1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_hir_typeck/src/upvar.rs#L375-L471>

Note that not every by-ref capture from the parent coroutine-closure results in a "lending" borrow. See the **Follow-up: When do async closures implement the regular `Fn*` traits?** section below for more details, since this intimately influences whether or not the coroutine-closure is allowed to implement the `Fn*` family of traits.

#### By-move body + `FnOnce` quirk

There are several situations where the closure upvar analysis ends up inferring upvars for the coroutine-closure's child coroutine that are too relaxed, and end up resulting in borrow-checker errors. This is best illustrated via examples. For example, given:

```rust
fn force_fnonce<T: async FnOnce()>(t: T) -> T { t }

let x = String::new();
let c = force_fnonce(async move || {
    println!("{x}");
});
```

`x` will be moved into the coroutine-closure, but the coroutine that is returned would only borrow `&x`. However, since `force_fnonce` forces the coroutine-closure to `AsyncFnOnce`, which is not *lending*, we must force the capture to happen by-move[^bm1].

Similarly:

```rust
let x = String::new();
let y = String::new();
let c = async move || {
    drop(y);
    println!("{x}");
};
```

`x` will be moved into the coroutine-closure, but the coroutine that is returned would only borrow `&x`. However, since we also capture `y` and drop it, the coroutine-closure is forced to be `AsyncFnOnce`. We must also force the capture of `x` to happen by-move. To determine this situation in particular, since unlike the last example the coroutine-kind's closure-kind has not yet been constrained, we must analyze the body of the coroutine-closure to see if how all of the upvars are used, to determine if they've been used in a way that is "consuming" -- i.e. that would force it to `FnOnce`[^bm2].

[^bm1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_hir_typeck/src/upvar.rs#L211-L248>

[^bm2]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_hir_typeck/src/upvar.rs#L532-L539>

#### Follow-up: When do async closures implement the regular `Fn*` traits?

Well, first of all, all async closures implement `FnOnce` since they can always be called *at least once*.

For `Fn`/`FnMut`, the detailed answer involves answering a related question: is the coroutine-closure lending? Because if it is, then it cannot implement the non-lending `Fn`/`FnMut` traits.

Determining when the coroutine-closure must *lend* its upvars is implemented in the `should_reborrow_from_env_of_parent_coroutine_closure` helper function[^u1]. Specifically, this needs to happen in two places:

[^u1]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_hir_typeck/src/upvar.rs#L1818-L1860>

1.  Are we borrowing data owned by the parent closure? We can determine if that is the case by checking if the parent capture is by-move, EXCEPT if we apply a deref projection, which means we're reborrowing a reference that we captured by-move.

```rust
let x = &1i32; // Let's call this lifetime `'1`.
let c = async move || {
    println!("{:?}", *x);
    // Even though the inner coroutine borrows by ref, we're only capturing `*x`,
    // not `x`, so the inner closure is allowed to reborrow the data for `'1`.
};
```

2. If a coroutine is mutably borrowing from a parent capture, then that mutable borrow cannot live for longer than either the parent *or* the borrow that we have on the original upvar. Therefore we always need to borrow the child capture with the lifetime of the parent coroutine-closure's env.

```rust
let mut x = 1i32;
let c = async || {
    x = 1;
    // The parent borrows `x` for some `&'1 mut i32`.
    // However, when we call `c()`, we implicitly autoref for the signature of
    // `AsyncFnMut::async_call_mut`. Let's call that lifetime `'call`. Since
    // the maximum that `&'call mut &'1 mut i32` can be reborrowed is `&'call mut i32`,
    // the inner coroutine should capture w/ the lifetime of the coroutine-closure.
};
```

If either of these cases apply, then we should capture the borrow with the lifetime of the parent coroutine-closure's env. Luckily, if this function is not correct, then the program is not unsound, since we still borrowck and validate the choices made from this function -- the only side-effect is that the user may receive unnecessary borrowck errors.

### Instance resolution

If a coroutine-closure has a closure-kind of `FnOnce`, then its `AsyncFnOnce::call_once` and `FnOnce::call_once` implementations resolve to the coroutine-closure's body[^res1], and the `Future::poll` of the coroutine that gets returned resolves to the body of the child closure.

[^res1]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_ty_utils/src/instance.rs#L351>

If a coroutine-closure has a closure-kind of `FnMut`/`Fn`, then the same applies to `AsyncFn` and the corresponding `Future` implementation of the coroutine that gets returned.[^res1] However, we use a MIR shim to generate the implementation of `AsyncFnOnce::call_once`/`FnOnce::call_once`[^res2], and `Fn::call`/`FnMut::call_mut` instances if they exist[^res3].

[^res2]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_ty_utils/src/instance.rs#L341-L349>

[^res3]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_ty_utils/src/instance.rs#L312-L326>

This is represented by the `ConstructCoroutineInClosureShim`[^i1]. The `receiver_by_ref` bool will be true if this is the instance of `Fn::call`/`FnMut::call_mut`.[^i2] The coroutine that all of these instances returns corresponds to the by-move body we will have synthesized by this point.[^i3]

[^i1]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_middle/src/ty/instance.rs#L129-L134>

[^i2]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_middle/src/ty/instance.rs#L136-L141>

[^i3]: <https://github.com/rust-lang/rust/blob/07cbbdd69363da97075650e9be24b78af0bcdd23/compiler/rustc_middle/src/ty/instance.rs#L841>

### Borrow-checking

It turns out that borrow-checking async closures is pretty straightforward. After adding a new `DefiningTy::CoroutineClosure`[^bck1] variant, and teaching borrowck how to generate the signature of the coroutine-closure[^bck2], borrowck proceeds totally fine.

One thing to note is that we don't borrow-check the synthetic body we make for by-move coroutines, since by construction (and the validity of the by-ref coroutine body it was derived from) it must be valid.

[^bck1]: <https://github.com/rust-lang/rust/blob/705cfe0e966399e061d64dd3661bfbc57553ed87/compiler/rustc_borrowck/src/universal_regions.rs#L110-L115>

[^bck2]: <https://github.com/rust-lang/rust/blob/7c7bb7dc017545db732f5cffec684bbaeae0a9a0/compiler/rustc_borrowck/src/universal_regions.rs#L743-L790>
