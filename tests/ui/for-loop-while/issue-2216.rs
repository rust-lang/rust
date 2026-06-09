//@ run-pass
#![allow(unreachable_code)]
pub fn main() {
    let mut x = 0;

    'foo: loop {
        'bar: loop {
            loop {
                if 1 == 2 {
                    break 'foo;
                }
                else {
                    break 'bar;
                }
            }
            continue 'foo;
        }
        x = 42;
        break;
    }

    println!("{}", x);
    assert_eq!(x, 42);
}
