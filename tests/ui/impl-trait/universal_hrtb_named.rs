//@ run-pass

fn hrtb(f: impl for<'a> Fn(&'a u32) -> &'a u32) -> u32 {
    f(&22) + f(&44)
}

fn main() {
    let sum = hrtb(|x| x);
    assert_eq!(sum, 22 + 44);
}
