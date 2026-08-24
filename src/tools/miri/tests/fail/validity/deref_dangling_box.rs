#[repr(C)]
struct S {
    a: (),
    b: i8,
}

fn main() {
    let mut x = Box::new(S { a: (), b: 0 });
    unsafe { (&raw mut x).cast::<usize>().write(16) };
    let _ = x.a; //~ERROR: must be dereferenceable for 1 byte, but got 0x10[noalloc] which is a dangling pointer
}
