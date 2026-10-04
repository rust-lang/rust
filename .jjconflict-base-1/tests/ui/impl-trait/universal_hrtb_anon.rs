//@ run-pass

fn hrtb(f: impl Fn(&u32) -> u32) -> u32 {
    f(&22) + f(&44)
}

fn main() {
    let sum = hrtb(|x| x * 2);
    assert_eq!(sum, 22*2 + 44*2);
}
