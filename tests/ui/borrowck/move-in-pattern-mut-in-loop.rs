//@ run-pass

fn main() {
    let mut x = 42_i32;
    let opt = Some(&mut x);
    for _ in 0..5 {
        if let Some(_x) = opt {}
    }
}
