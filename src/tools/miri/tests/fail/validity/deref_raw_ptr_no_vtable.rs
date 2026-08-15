struct S<Tail: ?Sized> {
    f: i32,
    #[allow(unused)]
    g: Tail,
}

fn main() {
    let x = S { f: 0, g: 0 };
    let mut ptr: *const S<dyn std::fmt::Debug> = &x;
    unsafe { (&raw mut ptr).cast::<usize>().add(1).write(0) };
    let _val = unsafe { &(*ptr).f }; //~ERROR: vtable
}
