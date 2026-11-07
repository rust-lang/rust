//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![crate_name = "bevy_ecs"]

// We currently special case bevy from erroring on incorrect implied bounds
// from normalization (issue #109628).
// Otherwise, we would expect this to hit that error.

pub trait WorldQuery {}
impl WorldQuery for &u8 {}

pub struct Query<Q: WorldQuery>(Q);

pub trait SystemParam {
    type State;
}
impl<Q: WorldQuery + 'static> SystemParam for Query<Q> {
    type State = ();
    // `Q: 'static` is required because we need the TypeId of Q ...
}

pub struct ParamSet<T: SystemParam>(T) where T::State: Sized;

fn handler<'a>(x: ParamSet<Query<&'a u8>>) {
    let _: ParamSet<_> = x;
}

fn ref_handler<'a>(_: &ParamSet<Query<&'a u8>>) {}

fn main() {}
