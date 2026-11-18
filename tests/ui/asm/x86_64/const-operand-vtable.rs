//@ only-x86_64
//@ only-linux
//@ needs-asm-support
//@ run-pass
#![feature(asm_const_ptr, ptr_metadata)]
use std::mem::MaybeUninit;

// Test an asm const operand being passed a vtable.

trait Trait {
    #[allow(unused)]
    extern "C" fn method(&self, a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }
}

struct Struct;

impl Trait for Struct {}

/// Get a pointer to the vtable from a `&dyn Trait`.
const fn vtable_of(x: &dyn Trait) -> *const () {
    let meta: std::ptr::DynMetadata<dyn Trait> = std::ptr::metadata(x);
    unsafe { std::mem::transmute(meta) }
}

const VTABLE: *const () = vtable_of(&Struct);

#[repr(C)]
struct ManualVTable {
    drop_in_place: Option<fn(_: *mut Struct)>,
    size: usize,
    align: usize,
    method: extern "C" fn(_: &Struct, _: i32, _: i32) -> i32,
}

fn read() -> ManualVTable {
    let drop_in_place: MaybeUninit<fn(_: *mut Struct)>;
    let size: usize;
    let align: usize;
    let method: extern "C" fn(_: &Struct, _: i32, _: i32) -> i32;

    unsafe {
        #[rustfmt::skip]
        std::arch::asm!(
            "lea {vtable}, [rip + {input}]",
            "mov {drop_in_place}, qword ptr [{vtable}]",
            "mov {size}, qword ptr [{vtable} + 8]",
            "mov {align}, qword ptr [{vtable} + 16]",
            "mov {method}, qword ptr [{vtable} + 24]",
            input = const VTABLE,
            vtable = out(reg) _,
            drop_in_place = out(reg) drop_in_place,
            size = out(reg) size,
            align = out(reg) align,
            method = out(reg) method,
        );

        let drop_in_place: Option<fn(_: *mut Struct)> =
            std::mem::transmute::<_, MaybeUninit<_>>(drop_in_place).assume_init();

        ManualVTable { drop_in_place, size, align, method }
    }
}

fn main() {
    let ManualVTable { drop_in_place, size, align, method } = read();

    assert!(drop_in_place.is_none());

    assert_eq!(size, 0);
    assert_eq!(align, 1);

    assert_eq!(method(&Struct, 1, 2), 3);
    assert_eq!(method(&Struct, 4, 5), 9);
}
