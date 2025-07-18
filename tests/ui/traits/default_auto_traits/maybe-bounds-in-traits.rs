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

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "legacy_receiver"]
trait LegacyReceiver {}
impl<T: ?Sized + ?Leak> LegacyReceiver for &T {}
impl<T: ?Sized> LegacyReceiver for &mut T {}

#[lang = "default_trait1"]
auto trait Leak {}

struct NonLeakS;
impl !Leak for NonLeakS {}
struct LeakS;

mod supertraits {
    use crate::*;

    trait MaybeLeakT1: ?Leak {}
    trait MaybeLeakT2 where Self: ?Leak {}

    impl MaybeLeakT1 for NonLeakS {}
    impl MaybeLeakT2 for NonLeakS {}
}

mod maybe_self_assoc_type {
    use crate::*;

    trait TestBase1<T: ?Sized> {}
    trait TestBase2<T: ?Leak + ?Sized> {}

    trait Test1<T> {
        type MaybeLeakSelf: TestBase1<Self> where Self: ?Leak;
        //~^ ERROR the trait bound `Self: Leak` is not satisfied
        type LeakSelf: TestBase1<Self>;
    }

    trait Test2<T> {
        type MaybeLeakSelf: TestBase2<Self> where Self: ?Leak;
        type LeakSelf: TestBase2<Self>;
    }

    trait Test3 {
        type Leak1 = LeakS;
        type Leak2 = NonLeakS;
        //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
    }

    trait Test4 {
        type MaybeLeak1: ?Leak = LeakS;
        type MaybeLeak2: ?Leak = NonLeakS;
    }

    trait Test5: ?Leak {
        // ok, because assoc types have implicit where Self: Leak
        type MaybeLeakSelf1: TestBase1<Self>;
        type MaybeLeakSelf2: TestBase2<Self>;
    }
}

mod maybe_self_assoc_const {
    use crate::*;

    const fn size_of<T: ?Sized>() -> usize {
        0
    }

    trait Trait {
        const CLeak: usize = size_of::<Self>();
        const CNonLeak: usize = size_of::<Self>() where Self: ?Leak;
        //~^ ERROR the trait bound `Self: Leak` is not satisfied
    }
}

mod methods {
    use crate::*;

    trait Trait {
        fn leak_foo(&self) {}
        fn maybe_leak_foo(&self) where Self: ?Leak {}
        fn mut_leak_foo(&mut self) {}
        // there is no relax bound on corresponding Receiver impl
        fn mut_maybe_leak_foo(&mut self) where Self: ?Leak {}
        //~^ ERROR `&mut Self` cannot be used as the type of `self` without the `arbitrary_self_types`
    }

    impl Trait for NonLeakS {}
    impl Trait for LeakS {}

    fn foo() {
        LeakS.leak_foo();
        LeakS.maybe_leak_foo();
        NonLeakS.leak_foo();
        //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
        NonLeakS.maybe_leak_foo();
    }
}

fn main() {}
