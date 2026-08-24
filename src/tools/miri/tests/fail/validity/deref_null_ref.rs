fn main() {
    let mut x = &();
    unsafe { (&raw mut x).cast::<usize>().write(0) };
    let _ = *x; //~ERROR: null reference
}
