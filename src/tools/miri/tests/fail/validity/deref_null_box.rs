fn main() {
    let mut x = Box::new(());
    unsafe { (&raw mut x).cast::<usize>().write(0) };
    let _ = *x; //~ERROR: null box
}
