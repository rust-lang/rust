//@ run-pass
fn get_third<T>(t: (T, T, T)) -> T { let (_, _, x) = t; return x; }

pub fn main() {
    println!("{}", get_third((1, 2, 3)));
    assert_eq!(get_third((1, 2, 3)), 3);
    assert_eq!(get_third((5u8, 6u8, 7u8)), 7u8);
}
