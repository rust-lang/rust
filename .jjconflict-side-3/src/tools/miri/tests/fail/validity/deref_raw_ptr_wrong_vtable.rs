struct S<Tail: ?Sized> {
    f: i32,
    #[allow(unused)]
    g: Tail,
}

fn main() {
    let x = S { f: 0, g: 0 };
    let ptr1: *const S<dyn std::fmt::Debug> = &x;
    let mut ptr2: *const S<dyn std::fmt::Display> = &x;
    // Give ptr2 the vtable from ptr1.
    unsafe { (&raw mut ptr2).copy_from(&raw const ptr1 as *const _, 1) };
    let _val = unsafe { &(*ptr2).f }; //~ERROR: vtable
}
