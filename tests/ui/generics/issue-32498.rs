//@ run-pass
#![allow(dead_code)]

// Making sure that no overflow occurs.

struct L<T> {
    n: Option<T>,
}
type L8<T> = L<L<L<L<L<L<L<L<T>>>>>>>>;
type L64<T> = L8<L8<L8<L8<T>>>>;

fn main() {
    use std::mem::size_of;
    assert_eq!(size_of::<L64<L64<()>>>(), 1);
    assert_eq!(size_of::<L<L64<L64<()>>>>(), 1);
}
