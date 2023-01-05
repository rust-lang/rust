// compile-flags: --edition=2021

#![feature(rustc_attrs)]

use core::any::Any;
use core::marker::PhantomData;

fn main() {
    test::<MaskedStorage<GenericComp<Pos>>>(make());
    //~^ ERROR evaluate(Binder(TraitPredicate(<MaskedStorage<GenericComp<Pos>> as std::marker::Sized>, polarity:Positive), [])) = Ok(EvaluatedToOk)
    //~| ERROR evaluate(Binder(TraitPredicate(<MaskedStorage<GenericComp<Pos>> as std::marker::Sized>, polarity:Positive), [])) = Ok(EvaluatedToOk)

    test::<MaskedStorage<GenericComp2<Pos>>>(make());
    //~^ ERROR evaluate(Binder(TraitPredicate(<MaskedStorage<GenericComp2<Pos>> as std::marker::Sized>, polarity:Positive), [])) = Ok(EvaluatedToOkModuloRegions)
    //~| ERROR evaluate(Binder(TraitPredicate(<MaskedStorage<GenericComp2<Pos>> as std::marker::Sized>, polarity:Positive), [])) = Ok(EvaluatedToOkModuloRegions)
}

#[rustc_evaluate_where_clauses]
fn test<T: Sized>(_: T) {}

fn make<T>() -> T {
    todo!()
}

struct DerefWrap<T>(T);

impl<T> core::ops::Deref for DerefWrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Storage<T, D> {
    phantom: PhantomData<(T, D)>,
}

type ReadStorage<T> = Storage<T, DerefWrap<MaskedStorage<T>>>;

pub trait Component {
    type Storage;
}

struct VecStorage;

struct Pos;

impl Component for Pos {
    type Storage = VecStorage;
}

struct GenericComp<T> {
    _t: T,
}

impl<T: 'static> Component for GenericComp<T> {
    type Storage = VecStorage;
}

struct GenericComp2<T> {
    _t: T,
}

impl<T: 'static> Component for GenericComp2<T> where for<'a> &'a bool: 'a {
    type Storage = VecStorage;
}

struct ReadData {
    pos_interpdata: ReadStorage<GenericComp<Pos>>,
}

trait System {
    type SystemData;

    fn run(data: Self::SystemData, any: Box<dyn Any>);
}

struct Sys;

impl System for Sys {
    type SystemData = (ReadData, ReadStorage<Pos>);

    fn run((data, pos): Self::SystemData, any: Box<dyn Any>) {
        <ReadStorage<GenericComp<Pos>> as SystemData>::setup(any);

        ParJoin::par_join((&pos, &data.pos_interpdata));
    }
}

trait ParJoin {
    fn par_join(self)
    where
        Self: Sized,
    {
    }
}

impl<'a, T, D> ParJoin for &'a Storage<T, D>
where
    T: Component,
    D: core::ops::Deref<Target = MaskedStorage<T>>,
    T::Storage: Sync,
{
}

impl<A, B> ParJoin for (A, B)
where
    A: ParJoin,
    B: ParJoin,
{
}

pub trait SystemData {
    fn setup(any: Box<dyn Any>);
}

impl<T: 'static> SystemData for ReadStorage<T>
where
    T: Component,
{
    fn setup(any: Box<dyn Any>) {
        let storage: &MaskedStorage<T> = any.downcast_ref().unwrap();

        <dyn Any as CastFrom<MaskedStorage<T>>>::cast(&storage);
    }
}

pub struct MaskedStorage<T: Component> {
    _inner: T::Storage,
}

pub unsafe trait CastFrom<T> {
    fn cast(t: &T) -> &Self;
}

unsafe impl<T> CastFrom<T> for dyn Any
where
    T: Any + 'static,
{
    fn cast(t: &T) -> &Self {
        t
    }
}
