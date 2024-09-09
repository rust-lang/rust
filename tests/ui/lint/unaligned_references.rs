//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::fmt::Debug;
use std::mem::ManuallyDrop;

#[repr(packed)]
pub struct Good {
    data: u64,
    ptr: &'static u64,
    data2: [u64; 2],
    aligned: [u8; 32],
}

#[repr(packed(2))]
pub struct Packed2 {
    x: u32,
    y: u16,
    z: u8,
}

trait Foo {
    fn evil(&self);
}

// Test for #108122
#[automatically_derived]
impl Foo for Packed2 {
    fn evil(&self) {
        unsafe {
            &self.x; //~ ERROR reference to packed field
        }
    }
}

// Test for #115396
fn packed_dyn() {
    #[repr(packed)]
    struct Unaligned<T: ?Sized>(ManuallyDrop<T>);

    let ref local = Unaligned(ManuallyDrop::new([3, 5, 8u64]));
    let foo: &Unaligned<dyn Debug> = &*local;
    println!("{:?}", &*foo.0); //~ ERROR reference to packed field
    let foo: &Unaligned<[u64]> = &*local;
    println!("{:?}", &*foo.0); //~ ERROR reference to packed field

    // Even if the actual alignment is 1, we cannot know that when looking at `dyn Debug.`
    let ref local = Unaligned(ManuallyDrop::new([3, 5, 8u8]));
    let foo: &Unaligned<dyn Debug> = &*local;
    println!("{:?}", &*foo.0); //~ ERROR reference to packed field
    // However, we *can* know the alignment when looking at a slice.
    let foo: &Unaligned<[u8]> = &*local;
    println!("{:?}", &*foo.0); // no error!
}

// Test for #115396
fn packed_slice_behind_alias() {
    trait Mirror {
        type Assoc: ?Sized;
    }
    impl<T: ?Sized> Mirror for T {
        type Assoc = T;
    }

    struct W<T: ?Sized>(<T as Mirror>::Assoc);

    #[repr(packed)]
    struct Unaligned<T: ?Sized>(ManuallyDrop<W<T>>);

    // Even if the actual alignment is 1, we cannot know that when looking at `dyn Debug.`
    let ref local: Unaligned<[_; 3]> = Unaligned(ManuallyDrop::new(W([3, 5, 8u8])));
    let foo: &Unaligned<[u8]> = local;
    let x = &foo.0; // Fine, since the tail of `foo` is `[_]`
}

fn main() {
    unsafe {
        let good = Good { data: 0, ptr: &0, data2: [0, 0], aligned: [0; 32] };

        let _ = &good.ptr; //~ ERROR reference to packed field
        let _ = &good.data; //~ ERROR reference to packed field
        // Error even when turned into raw pointer immediately.
        let _ = &good.data as *const _; //~ ERROR reference to packed field
        let _: *const _ = &good.data; //~ ERROR reference to packed field
        // Error on method call.
        let _ = good.data.clone(); //~ ERROR reference to packed field
        // Error for nested fields.
        let _ = &good.data2[0]; //~ ERROR reference to packed field

        let _ = &*good.ptr; // ok, behind a pointer
        let _ = &good.aligned; // ok, has align 1
        let _ = &good.aligned[2]; // ok, has align 1
    }

    unsafe {
        let packed2 = Packed2 { x: 0, y: 0, z: 0 };
        let _ = &packed2.x; //~ ERROR reference to packed field
        let _ = &packed2.y; // ok, has align 2 in packed(2) struct
        let _ = &packed2.z; // ok, has align 1
        packed2.evil();
    }

    unsafe {
        struct U16(u16);

        impl Drop for U16 {
            fn drop(&mut self) {
                println!("{:p}", self);
            }
        }

        struct HasDrop;

        impl Drop for HasDrop {
            fn drop(&mut self) {}
        }

        #[allow(unused)]
        struct Wrapper {
            a: U16,
            b: HasDrop,
        }
        #[allow(unused)]
        #[repr(packed(2))]
        struct Wrapper2 {
            a: U16,
            b: HasDrop,
        }

        // An outer struct with more restrictive packing than the inner struct -- make sure we
        // notice that!
        #[repr(packed)]
        struct Misalign<T>(u8, T);

        let m1 = Misalign(0, Wrapper { a: U16(10), b: HasDrop });
        let _ref = &m1.1.a; //~ ERROR reference to packed field

        let m2 = Misalign(0, Wrapper2 { a: U16(10), b: HasDrop });
        let _ref = &m2.1.a; //~ ERROR reference to packed field
    }
}
