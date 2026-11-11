#[repr(align(8))]
struct S {}

fn main() {
    let mut x = &S {};
    unsafe { (&raw mut x).cast::<usize>().write(1) };
    let _val = x as *const _; //~ERROR: unaligned reference
}
