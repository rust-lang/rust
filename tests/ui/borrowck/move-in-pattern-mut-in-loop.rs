// Regression test for #80913.

fn main() {
    let mut x = 42_i32;
    let mut opt = Some(&mut x);
    for _ in 0..5 {
        if let Some(mut _x) = opt {}
        //~^ ERROR: use of moved value
    }
}
