#[repr(align(8))]
struct S {
    f: (),
}

fn main() {
    let mut x = Box::new(S { f: () });
    unsafe { (&raw mut x).cast::<usize>().write(1) };
    let _ = &x.f; //~ERROR: unaligned box
}
