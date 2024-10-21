use std::cell::UnsafeCell;
use std::mem::transmute;

fn main() {
    // Array.
    let _a: [&mut u8; 2] = unsafe { transmute([&1u8; 2]) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell

    // Assert diagnostics show field names.
    {
        #[repr(transparent)]
        struct A {
            a: B,
        }
        #[repr(transparent)]
        struct B {
            b: C,
        }
        #[repr(transparent)]
        struct C {
            c: &'static D,
        }
        #[repr(transparent)]
        struct D {
            d: UnsafeCell<u8>,
        }

        #[repr(transparent)]
        struct E {
            e: F,
        }
        #[repr(transparent)]
        struct F {
            f: &'static G,
        }
        #[repr(transparent)]
        struct G {
            g: H,
        }
        #[repr(transparent)]
        struct H {
            h: u8,
        }

        let _: A = unsafe { transmute(&1u8) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
        let _: A = unsafe { transmute(E { e: F { f: &G { g: H { h: 0 } } } }) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
        let _: &'static UnsafeCell<u8> =
            unsafe { transmute(E { e: F { f: &G { g: H { h: 0 } } } }) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    }

    // Immutable to mutable reference.
    let _a: &mut u8 = unsafe { transmute(&1u8) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell

    // Immutable reference to `UnsafeCell`.
    let _a: &UnsafeCell<u8> = unsafe { transmute(&1u8) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior

    // Check in nested field.
    {
        #[repr(C)]
        struct Foo<T> {
            a: u32,
            b: Bar<T>,
        }
        #[repr(C)]
        struct Bar<T>(Baz<T>);
        #[repr(C)]
        struct Baz<T>(T);

        #[repr(C)]
        struct Other(&'static u8, &'static u8);

        let _: Foo<&'static mut u8> = unsafe { transmute(Other(&1u8, &1u8)) };
        //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
        let _: Foo<&'static UnsafeCell<u8>> = unsafe { transmute(Other(&1u8, &1u8)) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    }

    // Check that transmuting only part of the type to `UnsafeCell` triggers the lint.
    {
        #[repr(C)]
        struct A(u32);

        #[repr(C)]
        struct B(u16, UnsafeCell<u8>);

        #[repr(C)]
        struct C(u8, UnsafeCell<u8>);

        #[repr(C)]
        struct D(UnsafeCell<u32>);

        #[repr(C, packed)]
        struct E(u8, UnsafeCell<u16>, u8);

        #[repr(C, packed)]
        struct F(UnsafeCell<u16>, u16);

        let _: &B = unsafe { transmute(&A(0)) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
        let _: &D = unsafe { transmute(&C(0, UnsafeCell::new(0))) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
        let _: &F = unsafe { transmute(&E(0, UnsafeCell::new(0), 0)) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    }

    // Check that we report all error, since once cast may be intentional but another not,
    // especially considering that `&T` to `&UnsafeCell<T>` may be valid but to `&mut T` never is.
    {
        #[repr(C)]
        struct Foo(&'static u8, &'static u8);
        #[repr(C)]
        struct Bar(&'static UnsafeCell<u8>, &'static mut u8);

        let _a: Bar = unsafe { transmute(Foo(&0, &0)) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
        //~| ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
    }

    // `UnsafeCell` reference casting.
    {
        #[repr(C)]
        struct A {
            a: u64,
            b: u32,
            c: u32,
        }

        #[repr(C)]
        struct B {
            a: u64,
            b: UnsafeCell<u32>,
            c: u32,
        }

        let a = A { a: 0, b: 0, c: 0 };
        let _b = unsafe { &*(&a as *const A as *const B) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    }

    // Unsized.
    {
        #[repr(C)]
        struct A<T: ?Sized> {
            a: u32,
            b: T,
        }

        #[repr(C)]
        struct B {
            a: UnsafeCell<u32>,
            b: [u32],
        }

        let a = &A { a: 0, b: [0_u32, 0] } as &A<[u32]>;
        let _b = unsafe { &*(a as *const A<[u32]> as *const B) };
        //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    }
}
