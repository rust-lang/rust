#![feature(ptr_metadata)]
#![feature(portable_simd)]
#![warn(clippy::volatile_composites)]

use std::ptr::null_mut;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MyDevRegisters {
    baseaddr: usize,
    count: usize,
}

#[repr(transparent)]
struct Wrapper<T>((), T, ());

// Not to be confused with std::ptr::NonNull
struct NonNull<T>(T);

impl<T> NonNull<T> {
    fn write_volatile(&self, _arg: &T) {
        unimplemented!("Something entirely unrelated to std::ptr::NonNull");
    }
}

fn main() {
    let regs = MyDevRegisters {
        baseaddr: 0xabc123,
        count: 42,
    };

    const DEVICE_ADDR: *mut MyDevRegisters = 0xdead as *mut _;

    // Raw pointer methods
    unsafe {
        (&raw mut (*DEVICE_ADDR).baseaddr).write_volatile(regs.baseaddr); // OK
        (&raw mut (*DEVICE_ADDR).count).write_volatile(regs.count); // OK

        DEVICE_ADDR.write_volatile(regs);
        //~^ volatile_composites

        let _regs = MyDevRegisters {
            baseaddr: (&raw const (*DEVICE_ADDR).baseaddr).read_volatile(), // OK
            count: (&raw const (*DEVICE_ADDR).count).read_volatile(),       // OK
        };

        let _regs = DEVICE_ADDR.read_volatile();
        //~^ volatile_composites
    }

    // std::ptr functions
    unsafe {
        std::ptr::write_volatile(&raw mut (*DEVICE_ADDR).baseaddr, regs.baseaddr); // OK
        std::ptr::write_volatile(&raw mut (*DEVICE_ADDR).count, regs.count); // OK

        std::ptr::write_volatile(DEVICE_ADDR, regs);
        //~^ volatile_composites

        let _regs = MyDevRegisters {
            baseaddr: std::ptr::read_volatile(&raw const (*DEVICE_ADDR).baseaddr), // OK
            count: std::ptr::read_volatile(&raw const (*DEVICE_ADDR).count),       // OK
        };

        let _regs = std::ptr::read_volatile(DEVICE_ADDR);
        //~^ volatile_composites
    }

    // core::ptr functions
    unsafe {
        core::ptr::write_volatile(&raw mut (*DEVICE_ADDR).baseaddr, regs.baseaddr); // OK
        core::ptr::write_volatile(&raw mut (*DEVICE_ADDR).count, regs.count); // OK

        core::ptr::write_volatile(DEVICE_ADDR, regs);
        //~^ volatile_composites

        let _regs = MyDevRegisters {
            baseaddr: core::ptr::read_volatile(&raw const (*DEVICE_ADDR).baseaddr), // OK
            count: core::ptr::read_volatile(&raw const (*DEVICE_ADDR).count),       // OK
        };

        let _regs = core::ptr::read_volatile(DEVICE_ADDR);
        //~^ volatile_composites
    }

    // std::ptr::NonNull
    unsafe {
        let ptr = std::ptr::NonNull::new(DEVICE_ADDR).unwrap();

        ptr.write_volatile(regs);
        //~^ volatile_composites

        let _regs = ptr.read_volatile();
        //~^ volatile_composites
    }

    // Red herring
    {
        let thing = NonNull("hello".to_string());

        thing.write_volatile(&"goodbye".into()); // OK
    }

    // Zero size types OK
    unsafe {
        struct Empty;

        (0xdead as *mut Empty).write_volatile(Empty); // OK
        // Note that this is OK because Wrapper<Empty> is itself ZST, not because of the repr transparent
        // handling tested below.
        (0xdead as *mut Wrapper<Empty>).write_volatile(Wrapper((), Empty, ())); // OK
    }

    // Via repr transparent newtype
    unsafe {
        (0xdead as *mut Wrapper<usize>).write_volatile(Wrapper((), 123, ())); // OK
        (0xdead as *mut Wrapper<Wrapper<usize>>).write_volatile(Wrapper((), Wrapper((), 123, ()), ())); // OK

        (0xdead as *mut Wrapper<MyDevRegisters>).write_volatile(Wrapper((), MyDevRegisters::default(), ()));
        //~^ volatile_composites
    }

    // Plain type alias OK
    unsafe {
        type MyU64 = u64;

        (0xdead as *mut MyU64).write_volatile(123); // OK
    }

    // Wide pointers are not OK as data
    unsafe {
        let things: &[u32] = &[1, 2, 3];

        (0xdead as *mut *const u32).write_volatile(things.as_ptr()); // OK

        let wideptr: *const [u32] = std::ptr::from_raw_parts(things.as_ptr(), things.len());
        (0xdead as *mut *const [u32]).write_volatile(wideptr);
        //~^ volatile_composites
    }

    // Plain pointers and pointers with lifetimes are OK
    unsafe {
        let v: u32 = 123;
        let rv: &u32 = &v;

        (0xdead as *mut &u32).write_volatile(rv); // OK
    }

    // C-style enums are OK
    unsafe {
        // Bad: need some specific repr
        enum PlainEnum {
            A = 1,
            B = 2,
            C = 3,
        }

        (0xdead as *mut PlainEnum).write_volatile(PlainEnum::A);
        //~^ volatile_composites

        // OK
        #[repr(u32)]
        enum U32Enum {
            A = 1,
            B = 2,
            C = 3,
        }

        (0xdead as *mut U32Enum).write_volatile(U32Enum::A); // OK

        // OK
        #[repr(C)]
        enum CEnum {
            A = 1,
            B = 2,
            C = 3,
        }
        (0xdead as *mut CEnum).write_volatile(CEnum::A); // OK

        // Nope
        enum SumType {
            A(String),
            B(u32),
            C,
        }
        (0xdead as *mut SumType).write_volatile(SumType::C);
        //~^ volatile_composites

        // A repr on a complex sum type is not good enough
        #[repr(C)]
        enum ReprSumType {
            A(String),
            B(u32),
            C,
        }
        (0xdead as *mut ReprSumType).write_volatile(ReprSumType::C);
        //~^ volatile_composites
    }

    // SIMD is OK
    unsafe {
        (0xdead as *mut std::simd::u32x4).write_volatile(std::simd::u32x4::splat(1)); // OK
    }

    // Can't see through generic wrapper
    unsafe {
        do_device_write::<MyDevRegisters>(0xdead as *mut _, Default::default()); // OK
    }

    let mut s = String::from("foo");
    unsafe {
        std::ptr::write_volatile(&mut s, String::from("bar"));
        //~^ volatile_composites
    }
}

// Generic OK
unsafe fn do_device_write<T>(ptr: *mut T, v: T) {
    unsafe {
        ptr.write_volatile(v); // OK
    }
}
