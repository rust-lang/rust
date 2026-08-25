//@ known-bug: #138156

#![feature(generic_const_exprs)]

#[derive(Default)]
pub struct GenId<const IDX: usize>;

pub trait IndexTrait: Default {
    const IDX: usize;
}
pub trait ToplogyIndex {
    type Idx: IndexTrait;
}

#[derive(Default)]
pub struct Expression<T: ToplogyIndex> {
    pub data: T,
}

fn i<T: ToplogyIndex, const IDX0: usize, const IDX1: usize>(s: Expression<T>) ->
    Expression<GenId<{ IDX0 | IDX1 }>>
where
    GenId<{ IDX0 | IDX1 }>: ToplogyIndex,
{
    Expression::default()
}

pub fn sum<In: ToplogyIndex>(s: Expression<In>) -> Expression<In>
where
    [(); In::Idx::IDX]:,
{
    s
}

fn param_position<In: ToplogyIndex>(s: Expression<In>)
where
    GenId<{ 1 | 2 }>: ToplogyIndex,
{
    sum(i::<_, 1, 2>(s));
}

fn main() {}
