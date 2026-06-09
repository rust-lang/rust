use std::cell::RefCell;
use std::mem::ManuallyDrop;
use std::ops::Deref;

union U1 {
    a: u8,
}

union U2 {
    a: ManuallyDrop<String>,
}

union U3<T> {
    a: ManuallyDrop<T>,
}

union U4<T: Copy> {
    a: T,
}

union U5 {
    a: usize,
}

union URef {
    p: &'static mut i32,
}

union URefCell {
    // field that does not drop but is not `Copy`, either
    a: (ManuallyDrop<RefCell<i32>>, i32),
}

fn deref_union_field(mut u: URef) {
    // Not an assignment but an access to the union field!
    *(u.p) = 13; //~ ERROR access to union field is unsafe
}

union A {
    a: usize,
    b: &'static &'static B,
}

union B {
    c: usize,
}

fn raw_deref_union_field(mut u: URef) {
    // This is unsafe because we first dereference u.p (reading uninitialized memory)
    let _p = &raw const *(u.p); //~ ERROR access to union field is unsafe
}

fn assign_noncopy_union_field(mut u: URefCell) {
    u.a = (ManuallyDrop::new(RefCell::new(0)), 1); // OK (assignment does not drop)
    u.a.0 = ManuallyDrop::new(RefCell::new(0)); // OK (assignment does not drop)
    u.a.1 = 1; // OK
}

fn generic_noncopy<T: Default>() {
    let mut u3 = U3 { a: ManuallyDrop::new(T::default()) };
    u3.a = ManuallyDrop::new(T::default()); // OK (assignment does not drop)
    *u3.a = T::default(); //~ ERROR access to union field is unsafe
}

fn generic_copy<T: Copy + Default>() {
    let mut u3 = U3 { a: ManuallyDrop::new(T::default()) };
    u3.a = ManuallyDrop::new(T::default()); // OK
    *u3.a = T::default(); //~ ERROR access to union field is unsafe

    let mut u4 = U4 { a: T::default() };
    u4.a = T::default(); // OK
}

fn main() {
    let mut u1 = U1 { a: 10 }; // OK
    let a = u1.a; //~ ERROR access to union field is unsafe
    u1.a = 11; // OK

    let mut u2 = U1 { a: 10 };
    let a = &raw mut u2.a; // OK
    unsafe { *a = 3 };

    let mut u3 = U1 { a: 10 };
    let a = std::ptr::addr_of_mut!(u3.a); // OK
    unsafe { *a = 14 };

    let u4 = U5 { a: 2 };
    let vec = vec![1, 2, 3];
    // This is unsafe because we read u4.a (potentially uninitialized memory)
    // to use as an array index
    let _a = &raw const vec[u4.a]; //~ ERROR access to union field is unsafe

    let U1 { a } = u1; //~ ERROR access to union field is unsafe
    if let U1 { a: 12 } = u1 {} //~ ERROR access to union field is unsafe
    if let Some(U1 { a: 13 }) = Some(u1) {} //~ ERROR access to union field is unsafe
    // let U1 { .. } = u1; // OK

    let mut u2 = U2 { a: ManuallyDrop::new(String::from("old")) }; // OK
    u2.a = ManuallyDrop::new(String::from("new")); // OK (assignment does not drop)
    *u2.a = String::from("new"); //~ ERROR access to union field is unsafe

    let mut u3 = U3 { a: ManuallyDrop::new(0) }; // OK
    u3.a = ManuallyDrop::new(1); // OK
    *u3.a = 1; //~ ERROR access to union field is unsafe

    let mut u3 = U3 { a: ManuallyDrop::new(String::from("old")) }; // OK
    u3.a = ManuallyDrop::new(String::from("new")); // OK (assignment does not drop)
    *u3.a = String::from("new"); //~ ERROR access to union field is unsafe

    let mut unions = [U1 { a: 1 }, U1 { a: 2 }];

    // Array indexing + union field raw borrow - should be OK
    let ptr = &raw mut unions[0].a; // OK
    let ptr2 = &raw const unions[1].a; // OK

    let a = A { a: 0 };
    let _p = &raw const (**a.b).c; //~ ERROR access to union field is unsafe

    arbitrary_deref();
}

// regression test for https://github.com/rust-lang/rust/pull/141469#discussion_r2312546218
fn arbitrary_deref() {
    use std::ops::Deref;

    union A {
        a: usize,
        b: B,
    }

    #[derive(Copy, Clone)]
    struct B(&'static str);

    impl Deref for B {
        type Target = C;

        fn deref(&self) -> &C {
            println!("{:?}", self.0);
            &C { c: 0 }
        }
    }

    union C {
        c: usize,
    }

    let a = A { a: 0 };
    let _p = &raw const (*a.b).c; //~ ERROR access to union field is unsafe
}
