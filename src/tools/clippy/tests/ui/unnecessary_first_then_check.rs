#![warn(clippy::unnecessary_first_then_check)]
#![allow(clippy::useless_vec, clippy::const_is_empty)]

fn main() {
    let s = [1, 2, 3];
    let _: bool = s.first().is_some();
    let _: bool = s.first().is_none();

    let v = vec![1, 2, 3];
    let _: bool = v.first().is_some();

    let n = [[1, 2, 3], [4, 5, 6]];
    let _: bool = n[0].first().is_some();
    let _: bool = n[0].first().is_none();

    struct Foo {
        bar: &'static [i32],
    }
    let f = [Foo { bar: &[] }];
    let _: bool = f[0].bar.first().is_some();
    let _: bool = f[0].bar.first().is_none();
}
