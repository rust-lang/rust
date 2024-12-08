//@compile-flags: -Zmiri-permissive-provenance
#![feature(ptr_internals)]

fn main() {
    into_raw();
    into_unique();
    boxed_pair_to_vec();
}

fn into_raw() {
    unsafe {
        let b = Box::new(4i32);
        let r = Box::into_raw(b);

        // "lose the tag"
        let r2 = ((r as usize) + 0) as *mut i32;
        *(&mut *r2) = 7;

        // Use original ptr again
        *(&mut *r) = 17;
        drop(Box::from_raw(r));
    }
}

fn into_unique() {
    unsafe {
        let b = Box::new(4i32);
        let u = Box::into_unique(b).0;

        // "lose the tag"
        let r = ((u.as_ptr() as usize) + 0) as *mut i32;
        *(&mut *r) = 7;

        // Use original ptr again.
        drop(Box::from_raw(u.as_ptr()));
    }
}

fn boxed_pair_to_vec() {
    #[repr(C)]
    #[derive(Debug)]
    struct PairFoo {
        fst: Foo,
        snd: Foo,
    }

    #[derive(Debug)]
    struct Foo(#[allow(dead_code)] u64);
    fn reinterstruct(box_pair: Box<PairFoo>) -> Vec<Foo> {
        let ref_pair = Box::leak(box_pair) as *mut PairFoo;
        let ptr_foo = unsafe { std::ptr::addr_of_mut!((*ref_pair).fst) };
        unsafe { Vec::from_raw_parts(ptr_foo, 2, 2) }
    }

    let pair_foo = Box::new(PairFoo { fst: Foo(42), snd: Foo(1337) });
    println!("pair_foo = {:?}", pair_foo);
    for (n, foo) in reinterstruct(pair_foo).into_iter().enumerate() {
        println!("foo #{} = {:?}", n, foo);
    }
}
