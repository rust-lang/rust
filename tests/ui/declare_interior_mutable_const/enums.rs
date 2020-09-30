#![warn(clippy::declare_interior_mutable_const)]

use std::cell::Cell;
use std::sync::atomic::AtomicUsize;

enum OptionalCell {
    Unfrozen(Cell<bool>),
    Frozen,
}

// a constant with enums should be linted only when the used variant is unfrozen (#3962).
const UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(true)); //~ ERROR interior mutable
const FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

const fn unfrozen_variant() -> OptionalCell {
    OptionalCell::Unfrozen(Cell::new(false))
}

const fn frozen_variant() -> OptionalCell {
    OptionalCell::Frozen
}

const UNFROZEN_VARIANT_FROM_FN: OptionalCell = unfrozen_variant(); //~ ERROR interior mutable
const FROZEN_VARIANT_FROM_FN: OptionalCell = frozen_variant();

enum NestedInnermost {
    Unfrozen(AtomicUsize),
    Frozen,
}

struct NestedInner {
    inner: NestedInnermost,
}

enum NestedOuter {
    NestedInner(NestedInner),
    NotNested(usize),
}

struct NestedOutermost {
    outer: NestedOuter,
}

// a constant with enums should be linted according to its value, no matter how structs involve.
const NESTED_UNFROZEN_VARIANT: NestedOutermost = NestedOutermost {
    outer: NestedOuter::NestedInner(NestedInner {
        inner: NestedInnermost::Unfrozen(AtomicUsize::new(2)),
    }),
}; //~ ERROR interior mutable
const NESTED_FROZEN_VARIANT: NestedOutermost = NestedOutermost {
    outer: NestedOuter::NestedInner(NestedInner {
        inner: NestedInnermost::Frozen,
    }),
};

trait AssocConsts {
    // When there's no default value, lint it only according to its type.
    // Further details are on the corresponding code (`NonCopyConst::check_trait_item`).
    const TO_BE_UNFROZEN_VARIANT: OptionalCell; //~ ERROR interior mutable
    const TO_BE_FROZEN_VARIANT: OptionalCell; //~ ERROR interior mutable

    // Lint default values accordingly.
    const DEFAULTED_ON_UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(false)); //~ ERROR interior mutable
    const DEFAULTED_ON_FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;
}

// The lint doesn't trigger for an assoc constant in a trait impl with an unfrozen type even if it
// has enums. Further details are on the corresponding code in 'NonCopyConst::check_impl_item'.
impl AssocConsts for u64 {
    const TO_BE_UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(false));
    const TO_BE_FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

    // even if this sets an unfrozen variant, the lint ignores it.
    const DEFAULTED_ON_FROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Cell::new(false));
}

// At first, I thought I'd need to check every patterns in `trait.rs`; but, what matters
// here are values; and I think substituted generics at definitions won't appear in MIR.
trait AssocTypes {
    type ToBeUnfrozen;

    const TO_BE_UNFROZEN_VARIANT: Option<Self::ToBeUnfrozen>;
    const TO_BE_FROZEN_VARIANT: Option<Self::ToBeUnfrozen>;
}

impl AssocTypes for u64 {
    type ToBeUnfrozen = AtomicUsize;

    const TO_BE_UNFROZEN_VARIANT: Option<Self::ToBeUnfrozen> = Some(Self::ToBeUnfrozen::new(4)); //~ ERROR interior mutable
    const TO_BE_FROZEN_VARIANT: Option<Self::ToBeUnfrozen> = None;
}

// Use raw pointers since direct generics have a false negative at the type level.
enum BothOfCellAndGeneric<T> {
    Unfrozen(Cell<*const T>),
    Generic(*const T),
    Frozen(usize),
}

impl<T> BothOfCellAndGeneric<T> {
    const UNFROZEN_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Unfrozen(Cell::new(std::ptr::null())); //~ ERROR interior mutable

    // This is a false positive. The argument about this is on `is_value_unfrozen_raw`
    const GENERIC_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Generic(std::ptr::null()); //~ ERROR interior mutable

    const FROZEN_VARIANT: BothOfCellAndGeneric<T> = BothOfCellAndGeneric::Frozen(5);

    // This is what is likely to be a false negative when one tries to fix
    // the `GENERIC_VARIANT` false positive.
    const NO_ENUM: Cell<*const T> = Cell::new(std::ptr::null()); //~ ERROR interior mutable
}

// associated types here is basically the same as the one above.
trait BothOfCellAndGenericWithAssocType {
    type AssocType;

    const UNFROZEN_VARIANT: BothOfCellAndGeneric<Self::AssocType> =
        BothOfCellAndGeneric::Unfrozen(Cell::new(std::ptr::null())); //~ ERROR interior mutable
    const GENERIC_VARIANT: BothOfCellAndGeneric<Self::AssocType> = BothOfCellAndGeneric::Generic(std::ptr::null()); //~ ERROR interior mutable
    const FROZEN_VARIANT: BothOfCellAndGeneric<Self::AssocType> = BothOfCellAndGeneric::Frozen(5);
}

fn main() {}
