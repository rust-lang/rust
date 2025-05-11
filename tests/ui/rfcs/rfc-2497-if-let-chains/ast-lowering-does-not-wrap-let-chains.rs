//@ run-pass
//@ edition: 2024

#![allow(irrefutable_let_patterns)]

fn main() {
    let first = Some(1);
    let second = Some(2);
    let mut n = 0;
    if let x = first && let y = second && 1 == 1 {
        assert_eq!(x, first);
        assert_eq!(y, second);
        n = 1;
    }
    assert_eq!(n, 1);
}
