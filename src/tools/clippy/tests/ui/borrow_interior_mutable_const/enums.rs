//@aux-build:helper.rs

#![deny(clippy::borrow_interior_mutable_const)]
#![allow(clippy::declare_interior_mutable_const)]

// this file (mostly) replicates its `declare` counterpart. Please see it for more discussions.

extern crate helper;

use std::cell::Cell;
use std::sync::atomic::AtomicUsize;

enum OptionalCell {
    Unfrozen(Cell<bool>),
    Frozen,
}

const UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(true));
const FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

fn borrow_optional_cell() {
    let _ = &UNFROZEN_VARIANT; //~ ERROR: interior mutability
    let _ = &FROZEN_VARIANT;
}

trait AssocConsts {
    const TO_BE_UNFROZEN_VARIANT: OptionalCell;
    const TO_BE_FROZEN_VARIANT: OptionalCell;

    const DEFAULTED_ON_UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(false));
    const DEFAULTED_ON_FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

    fn function() {
        // This is the "suboptimal behavior" mentioned in `is_value_unfrozen`
        // caused by a similar reason to unfrozen types without any default values
        // get linted even if it has frozen variants'.
        let _ = &Self::TO_BE_FROZEN_VARIANT; //~ ERROR: interior mutability

        // The lint ignores default values because an impl of this trait can set
        // an unfrozen variant to `DEFAULTED_ON_FROZEN_VARIANT` and use the default impl for `function`.
        let _ = &Self::DEFAULTED_ON_FROZEN_VARIANT; //~ ERROR: interior mutability
    }
}

impl AssocConsts for u64 {
    const TO_BE_UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(false));
    const TO_BE_FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

    fn function() {
        let _ = &<Self as AssocConsts>::TO_BE_UNFROZEN_VARIANT; //~ ERROR: interior mutability
        let _ = &<Self as AssocConsts>::TO_BE_FROZEN_VARIANT;
        let _ = &Self::DEFAULTED_ON_UNFROZEN_VARIANT; //~ ERROR: interior mutability
        let _ = &Self::DEFAULTED_ON_FROZEN_VARIANT;
    }
}

trait AssocTypes {
    type ToBeUnfrozen;

    const TO_BE_UNFROZEN_VARIANT: Option<Self::ToBeUnfrozen>;
    const TO_BE_FROZEN_VARIANT: Option<Self::ToBeUnfrozen>;

    // there's no need to test here because it's the exactly same as `trait::AssocTypes`
    fn function();
}

impl AssocTypes for u64 {
    type ToBeUnfrozen = AtomicUsize;

    const TO_BE_UNFROZEN_VARIANT: Option<Self::ToBeUnfrozen> = Some(Self::ToBeUnfrozen::new(4));
    const TO_BE_FROZEN_VARIANT: Option<Self::ToBeUnfrozen> = None;

    fn function() {
        let _ = &<Self as AssocTypes>::TO_BE_UNFROZEN_VARIANT; //~ ERROR: interior mutability
        let _ = &<Self as AssocTypes>::TO_BE_FROZEN_VARIANT;
    }
}

enum BothOfCellAndGeneric<T> {
    Unfrozen(Cell<*const T>),
    Generic(*const T),
    Frozen(usize),
}

impl<T> BothOfCellAndGeneric<T> {
    const UNFROZEN_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Unfrozen(Cell::new(std::ptr::null()));
    const GENERIC_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Generic(std::ptr::null());
    const FROZEN_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Frozen(5);

    fn function() {
        let _ = &Self::UNFROZEN_VARIANT; //~ ERROR: interior mutability
        let _ = &Self::GENERIC_VARIANT; //~ ERROR: interior mutability
        let _ = &Self::FROZEN_VARIANT;
    }
}

fn main() {
    // constants defined in foreign crates
    let _ = &helper::WRAPPED_PRIVATE_UNFROZEN_VARIANT; //~ ERROR: interior mutability
    let _ = &helper::WRAPPED_PRIVATE_FROZEN_VARIANT;
}
