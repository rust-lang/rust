#![deny(clippy::declare_interior_mutable_const)]
#![allow(clippy::missing_const_for_thread_local)]

use core::cell::{Cell, RefCell, UnsafeCell};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;
use core::sync::atomic::AtomicUsize;

fn main() {}

const _: Cell<u32> = Cell::new(0);
const UNSAFE_CELL: UnsafeCell<u32> = UnsafeCell::new(0); //~ declare_interior_mutable_const
const REF_CELL: RefCell<u32> = RefCell::new(0); //~ declare_interior_mutable_const
const CELL: Cell<u32> = Cell::new(0); //~ declare_interior_mutable_const

// Constants can't contain pointers or references to type with interior mutability.
const fn make_ptr() -> *const Cell<u32> {
    ptr::null()
}
const PTR: *const Cell<u32> = make_ptr();

const fn casted_to_cell_ptr() -> *const Cell<u32> {
    const VALUE: u32 = 0;
    &VALUE as *const _ as *const Cell<u32>
}
const TRANSMUTED_PTR: *const Cell<u32> = casted_to_cell_ptr();

const CELL_TUPLE: (bool, Cell<u32>) = (true, Cell::new(0)); //~ declare_interior_mutable_const
const CELL_ARRAY: [Cell<u32>; 2] = [Cell::new(0), Cell::new(0)]; //~ declare_interior_mutable_const

const UNINIT_CELL: MaybeUninit<Cell<&'static ()>> = MaybeUninit::uninit();

struct CellStruct {
    x: u32,
    cell: Cell<u32>,
}
//~v declare_interior_mutable_const
const CELL_STRUCT: CellStruct = CellStruct {
    x: 0,
    cell: Cell::new(0),
};

enum CellEnum {
    Cell(Cell<u32>),
}
const CELL_ENUM: CellEnum = CellEnum::Cell(Cell::new(0)); //~ declare_interior_mutable_const

const NONE_CELL: Option<Cell<u32>> = None;
const SOME_CELL: Option<Cell<u32>> = Some(Cell::new(0)); //~ declare_interior_mutable_const

struct NestedCell([(Option<Cell<u32>>,); 1]);
const NONE_NESTED_CELL: NestedCell = NestedCell([(None,)]);
const SOME_NESTED_CELL: NestedCell = NestedCell([(Some(Cell::new(0)),)]); //~ declare_interior_mutable_const

union UnionCell {
    cell: ManuallyDrop<Cell<u32>>,
    x: u32,
}
//~v declare_interior_mutable_const
const UNION_CELL: UnionCell = UnionCell {
    cell: ManuallyDrop::new(Cell::new(0)),
};
// Access to either union field is valid so we have to be conservative here.
const UNION_U32: UnionCell = UnionCell { x: 0 }; //~ declare_interior_mutable_const

struct Assoc;
impl Assoc {
    const SELF: Self = Self;
    const CELL: Cell<u32> = Cell::new(0); //~ declare_interior_mutable_const
}

struct AssocCell(Cell<u32>);
impl AssocCell {
    const SELF: Self = Self(Cell::new(0)); //~ declare_interior_mutable_const
    const NONE_SELF: Option<Self> = None;
    const SOME_SELF: Option<Self> = Some(Self(Cell::new(0))); //~ declare_interior_mutable_const
}

trait ConstDefault {
    // May or may not be `Freeze`
    const DEFAULT: Self;
}
impl ConstDefault for u32 {
    const DEFAULT: Self = 0;
}
impl<T: ConstDefault> ConstDefault for Cell<T> {
    // Interior mutability is forced by the trait.
    const DEFAULT: Self = Cell::new(T::DEFAULT);
}
impl<T: ConstDefault> ConstDefault for Option<Cell<T>> {
    // Could have been `None`
    const DEFAULT: Self = Some(Cell::new(T::DEFAULT)); //~ declare_interior_mutable_const
}

enum GenericEnumCell<T> {
    Cell(Cell<T>),
    Other(T),
}
impl<T: ConstDefault> ConstDefault for GenericEnumCell<T> {
    const DEFAULT: Self = Self::Cell(Cell::new(T::DEFAULT)); //~ declare_interior_mutable_const
}
impl<T: ConstDefault> GenericEnumCell<T> {
    const CELL: Self = Self::DEFAULT; //~ declare_interior_mutable_const
    const CELL_BY_DEFAULT: Self = Self::Cell(Cell::DEFAULT); //~ declare_interior_mutable_const
    const OTHER: Self = Self::Other(T::DEFAULT);
    const FROM_OTHER: Self = Self::OTHER;
}

enum GenericNestedEnumCell<T> {
    GenericEnumCell(GenericEnumCell<T>),
    EnumCell(GenericEnumCell<u32>),
    Other(T),
}
impl<T: ConstDefault> GenericNestedEnumCell<T> {
    const GENERIC_OTHER: Self = Self::GenericEnumCell(GenericEnumCell::<T>::FROM_OTHER);
    const GENERIC_CELL: Self = Self::GenericEnumCell(GenericEnumCell::<T>::CELL); //~ declare_interior_mutable_const
    const ENUM_OTHER: Self = Self::EnumCell(GenericEnumCell::<u32>::FROM_OTHER);
    const ENUM_CELL: Self = Self::EnumCell(GenericEnumCell::<u32>::CELL); //~ declare_interior_mutable_const
}

trait CellTrait: ConstDefault + Sized {
    // Must be non-`Freeze` due to the type
    const CELL: Cell<Self>; //~ declare_interior_mutable_const
    // May be non-`Freeze`, but may not be
    const OPTION_CELL: Option<Cell<Self>>;
    // May get redefined by the impl, but the default is non-`Freeze`.
    const SOME_CELL: Option<Cell<Self>> = Some(Cell::new(Self::DEFAULT)); //~ declare_interior_mutable_const
    // May get redefined by the impl, but the default is `Freeze`.
    const NONE_CELL: Option<Cell<Self>> = None;
}

trait CellWithAssoc {
    type T;
    const DEFAULT: Self::T;
    // Must be non-`Freeze` due to the type
    const CELL: Cell<Self::T>; //~ declare_interior_mutable_const
    // May be non-`Freeze`, but may not be
    const OPTION_CELL: Option<Cell<Self::T>>;
    // May get redefined by the impl, but the default is non-`Freeze`.
    const SOME_CELL: Option<Cell<Self::T>> = Some(Cell::new(Self::DEFAULT)); //~ declare_interior_mutable_const
    // May get redefined by the impl, but the default is `Freeze`.
    const NONE_CELL: Option<Cell<Self::T>> = None;
}

impl CellWithAssoc for () {
    type T = u32;
    const DEFAULT: Self::T = 0;
    const CELL: Cell<Self::T> = Cell::new(0);
    const OPTION_CELL: Option<Cell<Self::T>> = None;
}

trait WithAssoc {
    type T;
    const VALUE: Self::T;
}

impl WithAssoc for u32 {
    type T = Cell<u32>;
    // The cell comes from the impl block, not the trait.
    const VALUE: Self::T = Cell::new(0); //~ declare_interior_mutable_const
}

trait WithLayeredAssoc {
    type T: WithAssoc;
    const VALUE: <Self::T as WithAssoc>::T;
}

impl WithLayeredAssoc for u32 {
    type T = u32;
    // The cell comes from the impl block, not the trait.
    const VALUE: <Self::T as WithAssoc>::T = Cell::new(0); //~ declare_interior_mutable_const
}

trait WithGenericAssoc {
    type T<U>;
    const VALUE: Self::T<u32>;
}

impl WithGenericAssoc for u32 {
    type T<U> = Cell<U>;
    const VALUE: Self::T<u32> = Cell::new(0); //~ declare_interior_mutable_const
}

trait WithGenericAssocCell {
    type T<U>;
    const VALUE: Self::T<Cell<u32>>;
}

impl WithGenericAssocCell for u32 {
    type T<U> = Option<U>;
    const VALUE: Self::T<Cell<u32>> = None;
}

impl WithGenericAssocCell for i32 {
    type T<U> = Option<U>;
    const VALUE: Self::T<Cell<u32>> = Some(Cell::new(0)); //~ declare_interior_mutable_const
}

thread_local!(static THREAD_LOCAL_CELL: Cell<u32> = const { Cell::new(0) });
thread_local!(static THREAD_LOCAL_CELL2: Cell<u32> = Cell::new(0));
