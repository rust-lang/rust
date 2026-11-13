fn main() {
    let mut x = &();
    unsafe { (&raw mut x).cast::<usize>().write(0) };
    let _val = x as *const _; //~ERROR: null reference
}
