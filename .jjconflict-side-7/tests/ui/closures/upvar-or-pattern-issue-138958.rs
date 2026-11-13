//@ edition:2024
//@ check-pass

pub fn f(x: (u32, u32)) {
    let _ = || {
        let ((0, a) | (a, _)) = x;
        a
    };
}

fn main() {}
