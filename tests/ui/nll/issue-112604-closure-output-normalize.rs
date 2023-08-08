//check-pass

use higher_kinded_types::*;
mod higher_kinded_types {
    pub(crate) trait HKT {
        type Of<'lt>;
    }

    pub(crate) trait WithLifetime<'lt> {
        type T;
    }

    impl<T: ?Sized + for<'any> WithLifetime<'any>> HKT for T {
        type Of<'lt> = <T as WithLifetime<'lt>>::T;
    }
}

trait Trait {
    type Gat<'lt>;
}

impl Trait for () {
    type Gat<'lt> = ();
}

/// Same as `Trait`, but using HKTs rather than GATs
trait HTrait {
    type Hat: ?Sized + HKT;
}

impl<T: Trait> HTrait for T {
    type Hat = dyn for<'lt> WithLifetime<'lt, T = T::Gat<'lt>>;
}

impl<Hat: ?Sized + HKT> Trait for Box<dyn '_ + HTrait<Hat = Hat>> {
    type Gat<'lt> = Hat::Of<'lt>;
}

fn existential() -> impl for<'a> Trait<Gat<'a> = ()> {}

fn dyn_hoops<T: Trait>(
    _: T,
) -> Box<dyn HTrait<Hat = dyn for<'a> WithLifetime<'a, T = T::Gat<'a>>>> {
    loop {}
}

fn main() {
    let _ = || -> _ { dyn_hoops(existential()) };
}
