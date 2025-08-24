//@ compile-flags: -Zexperimental-default-bounds

#![feature(
    auto_traits,
    associated_type_defaults,
    generic_const_items,
    lang_items,
    more_maybe_bounds,
    negative_impls,
    no_core,
    rustc_attrs
)]
#![allow(incomplete_features)]
#![no_std]
#![no_core]

#[lang = "copy"]
pub trait Copy: ?Leak {}

#[lang = "pointee_sized"]
trait PointeeSized: ?Leak {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized + ?Leak {}

#[lang = "sized"]
trait Sized: MetaSized + ?Leak {}

#[lang = "legacy_receiver"]
trait LegacyReceiver: ?Leak {}
impl<T: ?Sized + ?Leak> LegacyReceiver for &T {}
// Omit `T: ?Leak`.
impl<T: ?Sized> LegacyReceiver for &mut T {}

#[lang = "default_trait1"]
auto trait Leak {}

struct NonLeakS;
impl !Leak for NonLeakS {}
struct LeakS;

mod supertraits {
    use crate::*;

    trait MaybeLeak: ?Leak {}
    impl MaybeLeak for NonLeakS {}

    trait LeakT {}
    impl LeakT for NonLeakS {}
    //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
}

mod assoc_type_maybe_bounds {
    use crate::*;

    trait Test1 {
        type Leak1 = LeakS;
        type Leak2 = NonLeakS;
        //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
    }

    trait Test2 {
        type MaybeLeak1: ?Leak = LeakS;
        type MaybeLeak2: ?Leak = NonLeakS;
    }
}

mod methods {
    use crate::*;

    trait ReceiveCheck1: ?Leak {
        fn foo(&self) {}
    }

    trait ReceiveCheck2: ?Leak {
        // There is no `?Leak` bound on corresponding `LegacyReceiver` impl.
        fn mut_foo(&mut self) {}
        //~^ ERROR `&mut Self` cannot be used as the type of `self` without the `arbitrary_self_types`
    }
}

fn main() {}
