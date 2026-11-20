//@ check-pass

trait Trait
where
    for<'a> Self::Gat<'a>: OtherTrait,
    for<'a, 'b, 'c> <Self::Gat<'a> as OtherTrait>::OtherGat<'b>: HigherRanked<'c>,
{
    type Gat<'a>;
}

trait OtherTrait {
    type OtherGat<'b>;
}

trait HigherRanked<'c> {}

fn lower_ranked<T: for<'b, 'c> OtherTrait<OtherGat<'b>: HigherRanked<'c>>>() {}

fn higher_ranked<T: Trait>()
where
    for<'a> T::Gat<'a>: OtherTrait,
    for<'a, 'b, 'c> <T::Gat<'a> as OtherTrait>::OtherGat<'b>: HigherRanked<'c>,
{
}

fn test<T: Trait>() {
    lower_ranked::<T::Gat<'_>>();
    higher_ranked::<T>();
}

fn main() {}
