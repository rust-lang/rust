//@ check-pass

#![allow(dead_code)]

trait Structure<E>: Sized where E: Encoding {
    type RefTarget: ?Sized;
    type FfiPtr;
    unsafe fn borrow_from_ffi_ptr<'a>(ptr: Self::FfiPtr) -> Option<&'a Self::RefTarget>;
}

enum Slice {}

impl<E> Structure<E> for Slice where E: Encoding {
    type RefTarget = [E::Unit];
    type FfiPtr = (*const E::FfiUnit, usize);
    unsafe fn borrow_from_ffi_ptr<'a>(_ptr: Self::FfiPtr) -> Option<&'a Self::RefTarget> {
        panic!()
    }
}

trait Encoding {
    type Unit: Unit;
    type FfiUnit;
}

trait Unit {}

enum Utf16 {}

impl Encoding for Utf16 {
    type Unit = Utf16Unit;
    type FfiUnit = u16;
}

struct Utf16Unit(pub u16);

impl Unit for Utf16Unit {}

struct SUtf16Str {
    _data: <Slice as Structure<Utf16>>::RefTarget,
}

impl SUtf16Str {
    pub unsafe fn from_ptr<'a>(ptr: <Slice as Structure<Utf16>>::FfiPtr)
    -> Option<&'a Self> {
        std::mem::transmute::<Option<&[<Utf16 as Encoding>::Unit]>, _>(
            <Slice as Structure<Utf16>>::borrow_from_ffi_ptr(ptr))
    }
}

fn main() {
    const TEXT_U16: &'static [u16] = &[];
    let _ = unsafe { SUtf16Str::from_ptr((TEXT_U16.as_ptr(), TEXT_U16.len())).unwrap() };
}
