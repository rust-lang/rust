//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// The old solver doesn't verify depth when looking up cache.
// To avoid breakage, we evaluate with higher recursion limit in the next solver
// and emit an FCW for this.
// See the `recursion_depth_exceeding_limit` FCW.

#![recursion_limit = "8"]

trait Trait {
    fn anyone_can_call(&self);
}

trait HasAssoc {
    type Assoc;
}

struct W<T>(T);

impl<T: HasAssoc> HasAssoc for W<T> {
    type Assoc = T::Assoc;
}

impl HasAssoc for () {
    type Assoc = ();
}

impl Trait for () {
    fn anyone_can_call(&self) {}
}

fn foo() {
    // Insert a cache entry for the old solver. Without this, the old solver
    // also overflows.
    let a: <W<W<W<W<W<W<W<()>>>>>>> as HasAssoc>::Assoc = loop {};
    a.anyone_can_call();

    let b: <W<W<W<W<W<W<W<W<W<W<()>>>>>>>>>> as HasAssoc>::Assoc = loop {};
    //[next]~^ WARN: overflow evaluating the requirement `<W<W<W<W<W<W<W<_>>>>>>> as HasAssoc>::Assoc == _` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<W<_>>>>>>> as HasAssoc>::Assoc == _` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `W<W<W<W<W<W<W<W<W<W<()>>>>>>>>>>: HasAssoc` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<_>>>>>> as HasAssoc>::Assoc well-formed` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<W<_>>>>>>> as HasAssoc>::Assoc == _` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<_>>>>>> as HasAssoc>::Assoc well-formed` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<W<_>>>>>>> as HasAssoc>::Assoc == _` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `<W<W<W<W<W<W<_>>>>>> as HasAssoc>::Assoc well-formed` [recursion_depth_exceeding_limit]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // Force normalization when looking up methods and the self_ty is normalized to infer.
    b.anyone_can_call();
}

fn main() {}
