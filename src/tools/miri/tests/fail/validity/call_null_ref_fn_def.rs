fn foo() {}

fn main() {
    let mut f = &foo;
    unsafe { (&raw mut f).cast::<usize>().write(0) };
    f(); //~ERROR: null reference
}
