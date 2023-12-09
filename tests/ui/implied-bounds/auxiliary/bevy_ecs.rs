// Related to Bevy regression #118553

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

pub struct ParamSet<T: SystemParam>(T)
where
    T::State: Sized;
