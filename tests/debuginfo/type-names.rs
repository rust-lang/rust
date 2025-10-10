//@ ignore-lldb

//@ ignore-aarch64-pc-windows-msvc: Arm64 Windows cdb doesn't support JavaScript extensions.

// GDB changed the way that it formatted Foreign types
//@ min-gdb-version: 9.2

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ==================================================================================

// gdb-command:run

// STRUCTS
// gdb-command:whatis simple_struct
// gdb-check:type = type_names::Struct1

// gdb-command:whatis generic_struct1
// gdb-check:type = type_names::GenericStruct<type_names::mod1::Struct2, type_names::mod1::mod2::Struct3>

// gdb-command:whatis generic_struct2
// gdb-check:type = type_names::GenericStruct<type_names::Struct1, extern "system" fn(isize) -> usize>

// gdb-command:whatis mod_struct
// gdb-check:type = type_names::mod1::Struct2

// ENUMS
// gdb-command:whatis simple_enum_1
// gdb-check:type = type_names::Enum1

// gdb-command:whatis simple_enum_2
// gdb-check:type = type_names::Enum1

// gdb-command:whatis simple_enum_3
// gdb-check:type = type_names::mod1::Enum2

// gdb-command:whatis generic_enum_1
// gdb-check:type = type_names::mod1::mod2::Enum3<type_names::mod1::Struct2>

// gdb-command:whatis generic_enum_2
// gdb-check:type = type_names::mod1::mod2::Enum3<type_names::Struct1>

// TUPLES
// gdb-command:whatis tuple1
// gdb-check:type = (u32, type_names::Struct1, type_names::mod1::mod2::Enum3<type_names::mod1::Struct2>)

// gdb-command:whatis tuple2
// gdb-check:type = ((type_names::Struct1, type_names::mod1::mod2::Struct3), type_names::mod1::Enum2, char)

// BOX
// gdb-command:whatis box1
// gdb-check:type = (alloc::boxed::Box<f32, alloc::alloc::Global>, i32)

// gdb-command:whatis box2
// gdb-check:type = (alloc::boxed::Box<type_names::mod1::mod2::Enum3<f32>, alloc::alloc::Global>, i32)

// REFERENCES
// gdb-command:whatis ref1
// gdb-check:type = (&type_names::Struct1, i32)

// gdb-command:whatis ref2
// gdb-check:type = (&type_names::GenericStruct<char, type_names::Struct1>, i32)

// gdb-command:whatis mut_ref1
// gdb-check:type = (&mut type_names::Struct1, i32)

// gdb-command:whatis mut_ref2
// gdb-check:type = (&mut type_names::GenericStruct<type_names::mod1::Enum2, f64>, i32)

// RAW POINTERS
// gdb-command:whatis mut_ptr1
// gdb-check:type = (*mut type_names::Struct1, isize)

// gdb-command:whatis mut_ptr2
// gdb-check:type = (*mut isize, isize)

// gdb-command:whatis mut_ptr3
// gdb-check:type = (*mut type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)

// gdb-command:whatis const_ptr1
// gdb-check:type = (*const type_names::Struct1, isize)

// gdb-command:whatis const_ptr2
// gdb-check:type = (*const isize, isize)

// gdb-command:whatis const_ptr3
// gdb-check:type = (*const type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)

// VECTORS
// gdb-command:whatis fixed_size_vec1
// gdb-check:type = ([type_names::Struct1; 3], i16)

// gdb-command:whatis fixed_size_vec2
// gdb-check:type = ([usize; 3], i16)

// gdb-command:whatis slice1
// gdb-check:type = &[usize]

// gdb-command:whatis slice2
// gdb-check:type = &mut [type_names::mod1::Enum2]

// TRAITS
// gdb-command:whatis box_trait
// gdb-check:type = alloc::boxed::Box<dyn type_names::Trait1, alloc::alloc::Global>

// gdb-command:whatis ref_trait
// gdb-check:type = &dyn type_names::Trait1

// gdb-command:whatis mut_ref_trait
// gdb-check:type = &mut dyn type_names::Trait1

// gdb-command:whatis generic_box_trait
// gdb-check:type = alloc::boxed::Box<dyn type_names::Trait2<i32, type_names::mod1::Struct2>, alloc::alloc::Global>

// gdb-command:whatis generic_ref_trait
// gdb-check:type = &dyn type_names::Trait2<type_names::Struct1, type_names::Struct1>

// gdb-command:whatis generic_mut_ref_trait
// gdb-check:type = &mut dyn type_names::Trait2<type_names::mod1::mod2::Struct3, type_names::GenericStruct<usize, isize>>

// gdb-command:whatis no_principal_trait
// gdb-check:type = alloc::boxed::Box<(dyn core::marker::Send + core::marker::Sync), alloc::alloc::Global>

// gdb-command:whatis has_associated_type_trait
// gdb-check:type = &(dyn type_names::Trait3<u32, AssocType=isize> + core::marker::Send)

// gdb-command:whatis has_associated_type_but_no_generics_trait
// gdb-check:type = &dyn type_names::TraitNoGenericsButWithAssocType<Output=isize>

// BARE FUNCTIONS
// gdb-command:whatis rust_fn
// gdb-check:type = (fn(core::option::Option<isize>, core::option::Option<&type_names::mod1::Struct2>), usize)

// gdb-command:whatis extern_c_fn
// gdb-check:type = (extern "C" fn(isize), usize)

// gdb-command:whatis unsafe_fn
// gdb-check:type = (unsafe fn(core::result::Result<char, f64>), usize)

// gdb-command:whatis rust_fn_with_return_value
// gdb-check:type = (fn(f64) -> usize, usize)

// gdb-command:whatis extern_c_fn_with_return_value
// gdb-check:type = (extern "C" fn() -> type_names::Struct1, usize)

// gdb-command:whatis unsafe_fn_with_return_value
// gdb-check:type = (unsafe fn(type_names::GenericStruct<u16, u8>) -> type_names::mod1::Struct2, usize)

// gdb-command:whatis generic_function_int
// gdb-check:type = (fn(isize) -> isize, usize)

// gdb-command:whatis generic_function_struct3
// gdb-check:type = (fn(type_names::mod1::mod2::Struct3) -> type_names::mod1::mod2::Struct3, usize)

// gdb-command:whatis variadic_function
// gdb-check:type = (unsafe extern "C" fn(*const u8, ...) -> isize, usize)

// CLOSURES
// gdb-command:whatis closure1
// gdb-check:type = (type_names::main::{closure_env#0}, usize)

// gdb-command:whatis closure2
// gdb-check:type = (type_names::main::{closure_env#1}, usize)

// FOREIGN TYPES
// gdb-command:whatis foreign1
// gdb-check:type = *mut type_names::{extern#0}::ForeignType1

// gdb-command:whatis foreign2
// gdb-check:type = *mut type_names::mod1::{extern#0}::ForeignType2

// === CDB TESTS ==================================================================================

// Note: `/n` causes the wildcard matches to be sorted to avoid depending on order in PDB which
// can be arbitrary.

// cdb-command: g

// STRUCTS
// 0-sized structs appear to be optimized away in some cases, so only check the structs that do
// actually appear.
// cdb-command:dv /t /n *_struct

// ENUMS
// cdb-command:dv /t /n *_enum_*
// cdb-check:union enum2$<type_names::mod1::mod2::Enum3<type_names::mod1::Struct2> > generic_enum_1 = [...]
// cdb-check:union enum2$<type_names::mod1::mod2::Enum3<type_names::Struct1> > generic_enum_2 = [...]
// cdb-check:union enum2$<type_names::Enum1> simple_enum_1 = [...]
// cdb-check:union enum2$<type_names::Enum1> simple_enum_2 = [...]
// cdb-check:union enum2$<type_names::mod1::Enum2> simple_enum_3 = [...]

// TUPLES
// cdb-command:dv /t /n tuple*
// cdb-check:struct tuple$<u32,type_names::Struct1,enum2$<type_names::mod1::mod2::Enum3<type_names::mod1::Struct2> > > tuple1 = [...]
// cdb-check:struct tuple$<tuple$<type_names::Struct1,type_names::mod1::mod2::Struct3>,enum2$<type_names::mod1::Enum2>,char> tuple2 = [...]

// BOX
// cdb-command:dv /t /n box*
// cdb-check:struct tuple$<alloc::boxed::Box<f32,alloc::alloc::Global>,i32> box1 = [...]
// cdb-check:struct tuple$<alloc::boxed::Box<enum2$<type_names::mod1::mod2::Enum3<f32> >,alloc::alloc::Global>,i32> box2 = [...]

// REFERENCES
// cdb-command:dv /t /n *ref*
// cdb-check:struct tuple$<ref_mut$<type_names::Struct1>,i32> mut_ref1 = [...]
// cdb-check:struct tuple$<ref_mut$<type_names::GenericStruct<enum2$<type_names::mod1::Enum2>,f64> >,i32> mut_ref2 = [...]
// cdb-check:struct tuple$<ref$<type_names::Struct1>,i32> ref1 = [...]
// cdb-check:struct tuple$<ref$<type_names::GenericStruct<char,type_names::Struct1> >,i32> ref2 = [...]

// RAW POINTERS
// cdb-command:dv /t /n *_ptr*
// cdb-check:struct tuple$<ptr_const$<type_names::Struct1>,isize> const_ptr1 = [...]
// cdb-check:struct tuple$<ptr_const$<isize>,isize> const_ptr2 = [...]
// cdb-check:struct tuple$<ptr_const$<enum2$<type_names::mod1::mod2::Enum3<type_names::Struct1> > >,isize> const_ptr3 = [...]
// cdb-check:struct tuple$<ptr_mut$<type_names::Struct1>,isize> mut_ptr1 = [...]
// cdb-check:struct tuple$<ptr_mut$<isize>,isize> mut_ptr2 = [...]
// cdb-check:struct tuple$<ptr_mut$<enum2$<type_names::mod1::mod2::Enum3<type_names::Struct1> > >,isize> mut_ptr3 = [...]

// VECTORS
// cdb-command:dv /t /n *vec*
// cdb-check:struct tuple$<array$<type_names::Struct1,3>,i16> fixed_size_vec1 = [...]
// cdb-check:struct tuple$<array$<usize,3>,i16> fixed_size_vec2 = [...]
// cdb-check:struct alloc::vec::Vec<usize,alloc::alloc::Global> vec1 = [...]
// cdb-check:struct alloc::vec::Vec<enum2$<type_names::mod1::Enum2>,alloc::alloc::Global> vec2 = [...]
// cdb-command:dv /t /n slice*
// cdb-check:struct ref$<slice2$<usize> > slice1 = [...]
// cdb-check:struct ref_mut$<slice2$<enum2$<type_names::mod1::Enum2> > > slice2 = [...]

// TRAITS
// cdb-command:dv /t /n *_trait

// cdb-check:struct alloc::boxed::Box<dyn$<type_names::Trait1>,alloc::alloc::Global> box_trait = [...]
// cdb-check:struct alloc::boxed::Box<dyn$<type_names::Trait2<i32,type_names::mod1::Struct2> >,alloc::alloc::Global> generic_box_trait = [...]
// cdb-check:struct ref_mut$<dyn$<type_names::Trait2<type_names::mod1::mod2::Struct3,type_names::GenericStruct<usize,isize> > > > generic_mut_ref_trait = [...]
// cdb-check:struct ref$<dyn$<type_names::Trait2<type_names::Struct1,type_names::Struct1> > > generic_ref_trait = [...]
// cdb-check:struct ref$<dyn$<type_names::TraitNoGenericsButWithAssocType<assoc$<Output,isize> > > > has_associated_type_but_no_generics_trait = struct ref$<dyn$<type_names::TraitNoGenericsButWithAssocType<assoc$<Output,isize> > > >
// cdb-check:struct ref$<dyn$<type_names::Trait3<u32,assoc$<AssocType,isize> >,core::marker::Send> > has_associated_type_trait = struct ref$<dyn$<type_names::Trait3<u32,assoc$<AssocType,isize> >,core::marker::Send> >
// cdb-check:struct ref_mut$<dyn$<type_names::Trait1> > mut_ref_trait = [...]
// cdb-check:struct alloc::boxed::Box<dyn$<core::marker::Send,core::marker::Sync>,alloc::alloc::Global> no_principal_trait = [...]
// cdb-check:struct ref$<dyn$<type_names::Trait1> > ref_trait = [...]

// BARE FUNCTIONS
// cdb-command:dv /t /n *_fn*
// cdb-check:struct tuple$<void (*)(isize),usize> extern_c_fn = [...]
// cdb-check:struct tuple$<type_names::Struct1 (*)(),usize> extern_c_fn_with_return_value = [...]
// cdb-check:struct tuple$<void (*)(enum2$<core::option::Option<isize> >,enum2$<core::option::Option<ref$<type_names::mod1::Struct2> > >),usize> rust_fn = [...]
// cdb-check:struct tuple$<usize (*)(f64),usize> rust_fn_with_return_value = [...]
// cdb-check:struct tuple$<void (*)(enum2$<core::result::Result<char,f64> >),usize> unsafe_fn = [...]
// cdb-check:struct tuple$<type_names::mod1::Struct2 (*)(type_names::GenericStruct<u16,u8>),usize> unsafe_fn_with_return_value = [...]
// cdb-command:dv /t /n *_function*
// cdb-check:struct tuple$<isize (*)(isize),usize> generic_function_int = [...]
// cdb-check:struct tuple$<type_names::mod1::mod2::Struct3 (*)(type_names::mod1::mod2::Struct3),usize> generic_function_struct3 = [...]
// cdb-check:struct tuple$<isize (*)(ptr_const$<u8>, ...),usize> variadic_function = [...]
// cdb-command:dx Debugger.State.Scripts.@"type-names.cdb".Contents.getFunctionDetails("rust_fn")
// cdb-check:Return Type: void
// cdb-check:Parameter Types: enum2$<core::option::Option<isize> >,enum2$<core::option::Option<ref$<type_names::mod1::Struct2> > >
// cdb-command:dx Debugger.State.Scripts.@"type-names.cdb".Contents.getFunctionDetails("rust_fn_with_return_value")
// cdb-check:Return Type: usize
// cdb-check:Parameter Types: f64
// cdb-command:dx Debugger.State.Scripts.@"type-names.cdb".Contents.getFunctionDetails("extern_c_fn_with_return_value")
// cdb-check:Return Type: type_names::Struct1
// cdb-check:Parameter Types:

// CLOSURES
// cdb-command:dv /t /n closure*
// cdb-check:struct tuple$<type_names::main::closure_env$0,usize> closure1 = [...]
// cdb-check:struct tuple$<type_names::main::closure_env$1,usize> closure2 = [...]

// FOREIGN TYPES
// cdb-command:dv /t /n foreign*
// cdb-check:struct type_names::extern$0::ForeignType1 * foreign1 = [...]
// cdb-check:struct type_names::mod1::extern$0::ForeignType2 * foreign2 = [...]

#![allow(unused_variables)]
#![feature(extern_types)]

use std::marker::PhantomData;
use std::ptr;

use self::Enum1::{Variant1, Variant2};

pub struct Struct1;
struct GenericStruct<T1, T2>(PhantomData<(T1, T2)>);

enum Enum1 {
    Variant1,
    Variant2(isize),
}

extern "C" {
    type ForeignType1;
}

mod mod1 {
    pub struct Struct2;

    pub enum Enum2 {
        Variant1,
        Variant2(super::Struct1),
    }

    pub mod mod2 {
        pub use self::Enum3::{Variant1, Variant2};
        pub struct Struct3;

        pub enum Enum3<T> {
            Variant1,
            Variant2(T),
        }
    }

    extern "C" {
        pub type ForeignType2;
    }
}

trait Trait1 {
    fn dummy(&self) {}
}
trait Trait2<T1, T2> {
    fn dummy(&self, _: T1, _: T2) {}
}
trait Trait3<T> {
    type AssocType;
    fn dummy(&self) -> T {
        panic!()
    }
}
trait TraitNoGenericsButWithAssocType {
    type Output;
    fn foo(&self) -> Self::Output;
}

impl Trait1 for isize {}
impl<T1, T2> Trait2<T1, T2> for isize {}
impl<T> Trait3<T> for isize {
    type AssocType = isize;
}
impl TraitNoGenericsButWithAssocType for isize {
    type Output = isize;
    fn foo(&self) -> Self::Output {
        *self
    }
}

fn rust_fn(_: Option<isize>, _: Option<&mod1::Struct2>) {}
extern "C" fn extern_c_fn(_: isize) {}
unsafe fn unsafe_fn(_: Result<char, f64>) {}

fn rust_fn_with_return_value(_: f64) -> usize {
    4
}
extern "C" fn extern_c_fn_with_return_value() -> Struct1 {
    Struct1
}
unsafe fn unsafe_fn_with_return_value(_: GenericStruct<u16, u8>) -> mod1::Struct2 {
    mod1::Struct2
}

fn generic_function<T>(x: T) -> T {
    x
}

#[allow(improper_ctypes)]
extern "C" {
    fn printf(_: *const u8, ...) -> isize;
}

// In many of the cases below, the type that is actually under test is wrapped
// in a tuple, e.g., Box<T>, references, raw pointers, fixed-size vectors, ...
// This is because GDB will not print the type name from DWARF debuginfo for
// some kinds of types (pointers, arrays, functions, ...)
// Since tuples are structs as far as GDB is concerned, their name will be
// printed correctly, so the tests below just construct a tuple type that will
// then *contain* the type name that we want to see.
fn main() {
    // Structs
    let simple_struct = Struct1;
    let generic_struct1: GenericStruct<mod1::Struct2, mod1::mod2::Struct3> =
        GenericStruct(PhantomData);
    let generic_struct2: GenericStruct<Struct1, extern "system" fn(isize) -> usize> =
        GenericStruct(PhantomData);
    let mod_struct = mod1::Struct2;

    // Enums
    let simple_enum_1 = Variant1;
    let simple_enum_2 = Variant2(0);
    let simple_enum_3 = mod1::Enum2::Variant2(Struct1);

    let generic_enum_1: mod1::mod2::Enum3<mod1::Struct2> = mod1::mod2::Variant1;
    let generic_enum_2 = mod1::mod2::Variant2(Struct1);

    // Tuples
    let tuple1 = (8u32, Struct1, mod1::mod2::Variant2(mod1::Struct2));
    let tuple2 = ((Struct1, mod1::mod2::Struct3), mod1::Enum2::Variant1, 'x');

    // Box
    let box1 = (Box::new(1f32), 0i32);
    let box2 = (Box::new(mod1::mod2::Variant2(1f32)), 0i32);

    // References
    let ref1 = (&Struct1, 0i32);
    let ref2 = (&GenericStruct::<char, Struct1>(PhantomData), 0i32);

    let mut mut_struct1 = Struct1;
    let mut mut_generic_struct = GenericStruct::<mod1::Enum2, f64>(PhantomData);
    let mut_ref1 = (&mut mut_struct1, 0i32);
    let mut_ref2 = (&mut mut_generic_struct, 0i32);

    // Raw Pointers
    let mut_ptr1: (*mut Struct1, isize) = (ptr::null_mut(), 0);
    let mut_ptr2: (*mut isize, isize) = (ptr::null_mut(), 0);
    let mut_ptr3: (*mut mod1::mod2::Enum3<Struct1>, isize) = (ptr::null_mut(), 0);

    let const_ptr1: (*const Struct1, isize) = (ptr::null(), 0);
    let const_ptr2: (*const isize, isize) = (ptr::null(), 0);
    let const_ptr3: (*const mod1::mod2::Enum3<Struct1>, isize) = (ptr::null(), 0);

    // Vectors
    let fixed_size_vec1 = ([Struct1, Struct1, Struct1], 0i16);
    let fixed_size_vec2 = ([0_usize, 1, 2], 0i16);

    let vec1 = vec![0_usize, 2, 3];
    let slice1 = &*vec1;
    let mut vec2 = vec![mod1::Enum2::Variant2(Struct1)];
    let slice2 = &mut *vec2;

    // Trait Objects
    let box_trait = Box::new(0_isize) as Box<dyn Trait1>;
    let ref_trait = &0_isize as &dyn Trait1;
    let mut mut_int1 = 0_isize;
    let mut_ref_trait = (&mut mut_int1) as &mut dyn Trait1;
    let no_principal_trait = Box::new(0_isize) as Box<(dyn Send + Sync)>;
    let has_associated_type_trait = &0_isize as &(dyn Trait3<u32, AssocType = isize> + Send);
    let has_associated_type_but_no_generics_trait =
        &0_isize as &dyn TraitNoGenericsButWithAssocType<Output = isize>;

    let generic_box_trait = Box::new(0_isize) as Box<dyn Trait2<i32, mod1::Struct2>>;
    let generic_ref_trait = (&0_isize) as &dyn Trait2<Struct1, Struct1>;

    let mut generic_mut_ref_trait_impl = 0_isize;
    let generic_mut_ref_trait = (&mut generic_mut_ref_trait_impl)
        as &mut dyn Trait2<mod1::mod2::Struct3, GenericStruct<usize, isize>>;

    // Bare Functions
    let rust_fn = (rust_fn, 0_usize);
    let extern_c_fn = (extern_c_fn, 0_usize);
    let unsafe_fn = (unsafe_fn, 0_usize);

    let rust_fn_with_return_value = (rust_fn_with_return_value, 0_usize);
    let extern_c_fn_with_return_value = (extern_c_fn_with_return_value, 0_usize);
    let unsafe_fn_with_return_value = (unsafe_fn_with_return_value, 0_usize);

    let generic_function_int = (generic_function::<isize>, 0_usize);
    let generic_function_struct3 = (generic_function::<mod1::mod2::Struct3>, 0_usize);

    let variadic_function = (printf, 0_usize);

    // Closures
    // I (mw) am a bit unclear about the current state of closures, their
    // various forms (boxed, unboxed, proc, capture-by-ref, by-val, once) and
    // how that maps to rustc's internal representation of these forms.
    // Once closures have reached their 1.0 form, the tests below should
    // probably be expanded.
    let closure1 = (|x: isize| {}, 0_usize);
    let closure2 = (|x: i8, y: f32| (x as f32) + y, 0_usize);

    // Foreign Types
    let foreign1 = unsafe { 0 as *const ForeignType1 };
    let foreign2 = unsafe { 0 as *const mod1::ForeignType2 };

    zzz(); // #break
}

#[inline(never)]
fn zzz() {
    ()
}
