// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    use std::mem::replace;
    let mut x = 5;
    replace(&mut x, 6);
    {
        use std::mem::*;
        let mut y = 6;
        swap(&mut x, &mut y);
    }
}
