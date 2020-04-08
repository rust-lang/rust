pub fn dec_read_dec(i: &mut i32) -> i32 {
    *i -= 1;
    let ret = *i;
    *i -= 1;
    ret
}

pub fn minus_1(i: &i32) -> i32 {
    dec_read_dec(&mut i.clone())
}

fn main() {
    let mut i = 10;
    assert_eq!(minus_1(&i), 9);
    assert_eq!(i, 10);
    assert_eq!(dec_read_dec(&mut i), 9);
    assert_eq!(i, 8);
}
