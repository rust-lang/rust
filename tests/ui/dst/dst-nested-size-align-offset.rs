//@ run-pass
#![allow(dead_code)]

use std::cell::{Cell, UnsafeCell};
use std::mem::{
    ManuallyDrop, align_of, align_of_val, align_of_val_raw, offset_of, size_of, size_of_val,
    size_of_val_raw,
};
use std::ptr::{addr_of, slice_from_raw_parts};

// --- Slice-tailed DST hierarchy ---

#[repr(C)]
struct AlignedInner<T: ?Sized> {
    a: u32,
    b: u8,
    tail: T,
}

#[repr(C, packed(2))]
struct Packed2AroundAligned<T: ?Sized> {
    z: u8,
    inner: ManuallyDrop<AlignedInner<T>>,
}

#[repr(C, packed)]
struct Packed1AroundAligned<T: ?Sized> {
    z: u8,
    inner: ManuallyDrop<AlignedInner<T>>,
}

#[repr(C, align(8))]
struct Align8AroundPacked2<T: ?Sized> {
    w: u8,
    outer: Packed2AroundAligned<T>,
}

#[repr(C, packed(2))]
struct Packed2AroundAlign8<T: ?Sized> {
    v: u8,
    aligned: ManuallyDrop<Align8AroundPacked2<T>>,
}

#[repr(C)]
struct DeepReprC<T: ?Sized> {
    a: u64,
    b: (u32, (u16, AlignedInner<T>)),
}

macro_rules! check_slice_dst_len {
    ($len:expr, $elem:ty) => {{
        unsafe {
            let base = std::ptr::without_provenance::<()>(0x1000);
            let raw_slice = slice_from_raw_parts(base, $len);

            let p = raw_slice as *const AlignedInner<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<AlignedInner<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<AlignedInner<[$elem; $len]>>());
            assert_eq!(
                addr_of!((*p).tail).cast::<u8>().offset_from(p.cast::<u8>()) as usize,
                offset_of!(AlignedInner<[$elem; $len]>, tail),
            );

            let p = raw_slice as *const Packed2AroundAligned<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<Packed2AroundAligned<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<Packed2AroundAligned<[$elem; $len]>>());

            let p = raw_slice as *const Packed1AroundAligned<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<Packed1AroundAligned<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<Packed1AroundAligned<[$elem; $len]>>());

            let p = raw_slice as *const Align8AroundPacked2<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<Align8AroundPacked2<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<Align8AroundPacked2<[$elem; $len]>>());
            assert_eq!(
                addr_of!((*p).outer).cast::<u8>().offset_from(p.cast::<u8>()) as usize,
                offset_of!(Align8AroundPacked2<[$elem; $len]>, outer),
            );

            let p = raw_slice as *const Packed2AroundAlign8<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<Packed2AroundAlign8<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<Packed2AroundAlign8<[$elem; $len]>>());

            let p = raw_slice as *const DeepReprC<[$elem]>;
            assert_eq!(size_of_val_raw(p), size_of::<DeepReprC<[$elem; $len]>>());
            assert_eq!(align_of_val_raw(p), align_of::<DeepReprC<[$elem; $len]>>());
            assert_eq!(
                addr_of!((*p).b).cast::<u8>().offset_from(p.cast::<u8>()) as usize,
                offset_of!(DeepReprC<[$elem; $len]>, b),
            );
        }
    }};
}

macro_rules! check_slice_dst_elem {
    ($elem:ty) => {{
        check_slice_dst_len!(0, $elem);
        check_slice_dst_len!(1, $elem);
        check_slice_dst_len!(2, $elem);
        check_slice_dst_len!(3, $elem);
        check_slice_dst_len!(4, $elem);
        check_slice_dst_len!(5, $elem);
        check_slice_dst_len!(7, $elem);
        check_slice_dst_len!(8, $elem);
    }};
}

// --- `dyn Trait`-tailed DST hierarchy ---

#[repr(align(1))]
struct A1([u8; 3]);
#[repr(align(2))]
struct A2([u8; 6]);
#[repr(align(4))]
struct A4([u8; 12]);
#[repr(align(8))]
struct A8([u8; 24]);
#[repr(align(16))]
struct A16([u8; 16]);

#[repr(C)]
struct DynInner<T: ?Sized>(u8, ManuallyDrop<UnsafeCell<Cell<T>>>);

#[repr(C)]
struct DynOffset2<T: ?Sized>([u8; 2], ManuallyDrop<T>);

#[repr(C)]
struct DynOffset6<T: ?Sized>([u8; 6], ManuallyDrop<T>);

#[repr(C, packed(2))]
struct DynPacked2<T: ?Sized>(u8, ManuallyDrop<T>);

#[repr(C, packed(4))]
struct DynPacked4Offset2<T: ?Sized>([u8; 2], ManuallyDrop<T>);

#[repr(C, packed(4))]
struct DynPacked4Offset3<T: ?Sized>([u8; 3], ManuallyDrop<T>);

#[repr(C, packed(2))]
struct DynPacked2AroundInner<T: ?Sized>(u8, ManuallyDrop<DynInner<T>>);

#[repr(C, align(8))]
struct DynAlign8AroundPacked2<T: ?Sized>(u8, DynPacked2AroundInner<T>);

macro_rules! check_dyn_dst {
    ($val:expr, $ty:ty) => {{
        let md = ManuallyDrop::new($val);
        let md_sized: &ManuallyDrop<$ty> = &md;
        let md_dyn: &ManuallyDrop<dyn Send> = md_sized;
        assert_eq!(size_of_val(md_sized), size_of_val(md_dyn));
        assert_eq!(align_of_val(md_sized), align_of_val(md_dyn));

        let inner = DynInner(1, ManuallyDrop::new(UnsafeCell::new(Cell::new($val))));
        let inner_sized: &DynInner<$ty> = &inner;
        let inner_dyn: &DynInner<dyn Send> = inner_sized;
        assert_eq!(size_of_val(inner_sized), size_of_val(inner_dyn));
        assert_eq!(align_of_val(inner_sized), align_of_val(inner_dyn));
        assert_eq!(addr_of!(inner_sized.1).cast::<u8>(), addr_of!(inner_dyn.1).cast::<u8>());

        let o2 = DynOffset2([1, 2], ManuallyDrop::new($val));
        let o2_sized: &DynOffset2<$ty> = &o2;
        let o2_dyn: &DynOffset2<dyn Send> = o2_sized;
        assert_eq!(size_of_val(o2_sized), size_of_val(o2_dyn));
        assert_eq!(align_of_val(o2_sized), align_of_val(o2_dyn));
        assert_eq!(addr_of!(o2_sized.1).cast::<u8>(), addr_of!(o2_dyn.1).cast::<u8>());

        let o6 = DynOffset6([1, 2, 3, 4, 5, 6], ManuallyDrop::new($val));
        let o6_sized: &DynOffset6<$ty> = &o6;
        let o6_dyn: &DynOffset6<dyn Send> = o6_sized;
        assert_eq!(size_of_val(o6_sized), size_of_val(o6_dyn));
        assert_eq!(align_of_val(o6_sized), align_of_val(o6_dyn));
        assert_eq!(addr_of!(o6_sized.1).cast::<u8>(), addr_of!(o6_dyn.1).cast::<u8>());

        let p2 = DynPacked2(1, ManuallyDrop::new($val));
        let p2_sized: &DynPacked2<$ty> = &p2;
        let p2_dyn: &DynPacked2<dyn Send> = p2_sized;
        assert_eq!(size_of_val(p2_sized), size_of_val(p2_dyn));
        assert_eq!(align_of_val(p2_sized), align_of_val(p2_dyn));
        assert_eq!(addr_of!(p2_sized.1).cast::<u8>(), addr_of!(p2_dyn.1).cast::<u8>());

        let p4_2 = DynPacked4Offset2([1, 2], ManuallyDrop::new($val));
        let p4_2_sized: &DynPacked4Offset2<$ty> = &p4_2;
        let p4_2_dyn: &DynPacked4Offset2<dyn Send> = p4_2_sized;
        assert_eq!(size_of_val(p4_2_sized), size_of_val(p4_2_dyn));
        assert_eq!(align_of_val(p4_2_sized), align_of_val(p4_2_dyn));
        assert_eq!(addr_of!(p4_2_sized.1).cast::<u8>(), addr_of!(p4_2_dyn.1).cast::<u8>());

        let p4_3 = DynPacked4Offset3([1, 2, 3], ManuallyDrop::new($val));
        let p4_3_sized: &DynPacked4Offset3<$ty> = &p4_3;
        let p4_3_dyn: &DynPacked4Offset3<dyn Send> = p4_3_sized;
        assert_eq!(size_of_val(p4_3_sized), size_of_val(p4_3_dyn));
        assert_eq!(align_of_val(p4_3_sized), align_of_val(p4_3_dyn));
        assert_eq!(addr_of!(p4_3_sized.1).cast::<u8>(), addr_of!(p4_3_dyn.1).cast::<u8>());

        let a8 = DynAlign8AroundPacked2(
            1,
            DynPacked2AroundInner(
                2,
                ManuallyDrop::new(DynInner(3, ManuallyDrop::new(UnsafeCell::new(Cell::new($val))))),
            ),
        );
        let a8_sized: &DynAlign8AroundPacked2<$ty> = &a8;
        let a8_dyn: &DynAlign8AroundPacked2<dyn Send> = a8_sized;
        assert_eq!(size_of_val(a8_sized), size_of_val(a8_dyn));
        assert_eq!(align_of_val(a8_sized), align_of_val(a8_dyn));
        assert_eq!(addr_of!(a8_sized.1).cast::<u8>(), addr_of!(a8_dyn.1).cast::<u8>());
    }};
}

fn main() {
    check_slice_dst_elem!(());
    check_slice_dst_elem!(u8);
    check_slice_dst_elem!(u16);
    check_slice_dst_elem!([u8; 3]);
    check_slice_dst_elem!(u32);
    check_slice_dst_elem!([u16; 3]);
    check_slice_dst_elem!(u64);

    check_dyn_dst!((), ());
    check_dyn_dst!(A1([0; 3]), A1);
    check_dyn_dst!(A2([0; 6]), A2);
    check_dyn_dst!(A4([0; 12]), A4);
    check_dyn_dst!(A8([0; 24]), A8);
    check_dyn_dst!(A16([0; 16]), A16);
}
