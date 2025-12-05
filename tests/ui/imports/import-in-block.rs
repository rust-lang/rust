//@ run-pass

pub fn main() {
    use std::mem::replace;
    let mut x = 5;
    let _ = replace(&mut x, 6);
    {
        use std::mem::*;
        let mut y = 6;
        swap(&mut x, &mut y);
    }
}
