//@ build-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

// Regression test for #157729.
// The object candidates for `dyn Derived<()>` can differ only in the structural
// representation of semantically equivalent next-gen region constraints. These
// constraints must be normalized before response canonicalization so the
// candidates merge instead of remaining ambiguous and causing an ICE during
// instance resolution.

trait Proj {
    type S;
}

impl Proj for () {
    type S = ();
}

impl Proj for i32 {
    type S = i32;
}

trait Base<T> {
    fn is_base(&self);
}

trait Derived<B: Proj>: Base<B::S> + Base<()> {
    fn is_derived(&self);
}

fn f<P: Proj>(obj: &dyn Derived<P>) {
    obj.is_derived();
    Base::<P::S>::is_base(obj);
    Base::<()>::is_base(obj);
}

fn main() {
    let _x: fn(_) = f::<()>;
    let _x: fn(_) = f::<i32>;
}
