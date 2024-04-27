#![deny(clippy::borrow_interior_mutable_const)]
#![allow(clippy::declare_interior_mutable_const)]

// this file replicates its `declare` counterpart. Please see it for more discussions.

use std::borrow::Cow;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

trait ConcreteTypes {
    const ATOMIC: AtomicUsize;
    const STRING: String;

    fn function() {
        let _ = &Self::ATOMIC; //~ ERROR: interior mutability
        let _ = &Self::STRING;
    }
}

impl ConcreteTypes for u64 {
    const ATOMIC: AtomicUsize = AtomicUsize::new(9);
    const STRING: String = String::new();

    fn function() {
        // Lint this again since implementers can choose not to borrow it.
        let _ = &Self::ATOMIC; //~ ERROR: interior mutability
        let _ = &Self::STRING;
    }
}

// a helper trait used below
trait ConstDefault {
    const DEFAULT: Self;
}

trait GenericTypes<T, U> {
    const TO_REMAIN_GENERIC: T;
    const TO_BE_CONCRETE: U;

    fn function() {
        let _ = &Self::TO_REMAIN_GENERIC;
    }
}

impl<T: ConstDefault> GenericTypes<T, AtomicUsize> for Vec<T> {
    const TO_REMAIN_GENERIC: T = T::DEFAULT;
    const TO_BE_CONCRETE: AtomicUsize = AtomicUsize::new(11);

    fn function() {
        let _ = &Self::TO_REMAIN_GENERIC;
        let _ = &Self::TO_BE_CONCRETE; //~ ERROR: interior mutability
    }
}

// a helper type used below
pub struct Wrapper<T>(T);

trait AssocTypes {
    type ToBeFrozen;
    type ToBeUnfrozen;
    type ToBeGenericParam;

    const TO_BE_FROZEN: Self::ToBeFrozen;
    const TO_BE_UNFROZEN: Self::ToBeUnfrozen;
    const WRAPPED_TO_BE_UNFROZEN: Wrapper<Self::ToBeUnfrozen>;
    const WRAPPED_TO_BE_GENERIC_PARAM: Wrapper<Self::ToBeGenericParam>;

    fn function() {
        let _ = &Self::TO_BE_FROZEN;
        let _ = &Self::WRAPPED_TO_BE_UNFROZEN;
    }
}

impl<T: ConstDefault> AssocTypes for Vec<T> {
    type ToBeFrozen = u16;
    type ToBeUnfrozen = AtomicUsize;
    type ToBeGenericParam = T;

    const TO_BE_FROZEN: Self::ToBeFrozen = 12;
    const TO_BE_UNFROZEN: Self::ToBeUnfrozen = AtomicUsize::new(13);
    const WRAPPED_TO_BE_UNFROZEN: Wrapper<Self::ToBeUnfrozen> = Wrapper(AtomicUsize::new(14));
    const WRAPPED_TO_BE_GENERIC_PARAM: Wrapper<Self::ToBeGenericParam> = Wrapper(T::DEFAULT);

    fn function() {
        let _ = &Self::TO_BE_FROZEN;
        let _ = &Self::TO_BE_UNFROZEN; //~ ERROR: interior mutability
        let _ = &Self::WRAPPED_TO_BE_UNFROZEN; //~ ERROR: interior mutability
        let _ = &Self::WRAPPED_TO_BE_GENERIC_PARAM;
    }
}

// a helper trait used below
trait AssocTypesHelper {
    type NotToBeBounded;
    type ToBeBounded;

    const NOT_TO_BE_BOUNDED: Self::NotToBeBounded;
}

trait AssocTypesFromGenericParam<T>
where
    T: AssocTypesHelper<ToBeBounded = AtomicUsize>,
{
    const NOT_BOUNDED: T::NotToBeBounded;
    const BOUNDED: T::ToBeBounded;

    fn function() {
        let _ = &Self::NOT_BOUNDED;
        let _ = &Self::BOUNDED; //~ ERROR: interior mutability
    }
}

impl<T> AssocTypesFromGenericParam<T> for Vec<T>
where
    T: AssocTypesHelper<ToBeBounded = AtomicUsize>,
{
    const NOT_BOUNDED: T::NotToBeBounded = T::NOT_TO_BE_BOUNDED;
    const BOUNDED: T::ToBeBounded = AtomicUsize::new(15);

    fn function() {
        let _ = &Self::NOT_BOUNDED;
        let _ = &Self::BOUNDED; //~ ERROR: interior mutability
    }
}

trait SelfType: Sized {
    const SELF: Self;
    const WRAPPED_SELF: Option<Self>;

    fn function() {
        let _ = &Self::SELF;
        let _ = &Self::WRAPPED_SELF;
    }
}

impl SelfType for u64 {
    const SELF: Self = 16;
    const WRAPPED_SELF: Option<Self> = Some(20);

    fn function() {
        let _ = &Self::SELF;
        let _ = &Self::WRAPPED_SELF;
    }
}

impl SelfType for AtomicUsize {
    const SELF: Self = AtomicUsize::new(17);
    const WRAPPED_SELF: Option<Self> = Some(AtomicUsize::new(21));

    fn function() {
        let _ = &Self::SELF; //~ ERROR: interior mutability
        let _ = &Self::WRAPPED_SELF; //~ ERROR: interior mutability
    }
}

trait BothOfCellAndGeneric<T> {
    const DIRECT: Cell<T>;
    const INDIRECT: Cell<*const T>;

    fn function() {
        let _ = &Self::DIRECT;
        let _ = &Self::INDIRECT; //~ ERROR: interior mutability
    }
}

impl<T: ConstDefault> BothOfCellAndGeneric<T> for Vec<T> {
    const DIRECT: Cell<T> = Cell::new(T::DEFAULT);
    const INDIRECT: Cell<*const T> = Cell::new(std::ptr::null());

    fn function() {
        let _ = &Self::DIRECT;
        let _ = &Self::INDIRECT; //~ ERROR: interior mutability
    }
}

struct Local<T>(T);

impl<T> Local<T>
where
    T: ConstDefault + AssocTypesHelper<ToBeBounded = AtomicUsize>,
{
    const ATOMIC: AtomicUsize = AtomicUsize::new(18);
    const COW: Cow<'static, str> = Cow::Borrowed("tuvwxy");

    const GENERIC_TYPE: T = T::DEFAULT;

    const ASSOC_TYPE: T::NotToBeBounded = T::NOT_TO_BE_BOUNDED;
    const BOUNDED_ASSOC_TYPE: T::ToBeBounded = AtomicUsize::new(19);

    fn function() {
        let _ = &Self::ATOMIC; //~ ERROR: interior mutability
        let _ = &Self::COW;
        let _ = &Self::GENERIC_TYPE;
        let _ = &Self::ASSOC_TYPE;
        let _ = &Self::BOUNDED_ASSOC_TYPE; //~ ERROR: interior mutability
    }
}

fn main() {
    u64::ATOMIC.store(5, Ordering::SeqCst); //~ ERROR: interior mutability
    assert_eq!(u64::ATOMIC.load(Ordering::SeqCst), 9); //~ ERROR: interior mutability
}
