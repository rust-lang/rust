//[old]~ ERROR overflow evaluating the requirement `<Family as DistributionFamily>::Distribution<_> == _` [E0275]
//@ revisions: old next
//@[next] compile-flags: -Znext-solver


pub trait DistributionFamily {
    type Distribution<T>: Distribution<T, Family = Self>;
}

pub trait Distribution<Value> {
    type Family: DistributionFamily;

    fn single_value() -> Self;

    fn cartesian_product<T, U>(
        self,
        other: <Self::Family as DistributionFamily>::Distribution<T>,
    ) -> <Self::Family as DistributionFamily>::Distribution<U>;

    fn flatten<T>(self) -> <Self::Family as DistributionFamily>::Distribution<T>
    where
        Value: Distribution<T>;
}


fn start_event<Family: DistributionFamily>() -> Family::Distribution<u64> {
    Family::Distribution::single_value()
        .cartesian_product(Family::Distribution::single_value())
        //[next]~^ ERROR type annotations needed
        .flatten::<u64>()
}

fn main() {}
