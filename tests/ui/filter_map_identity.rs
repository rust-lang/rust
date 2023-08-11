#![allow(unused_imports, clippy::needless_return)]
#![warn(clippy::filter_map_identity)]

fn main() {
    let iterator = vec![Some(1), None, Some(2)].into_iter();
    let _ = iterator.filter_map(|x| x);

    let iterator = vec![Some(1), None, Some(2)].into_iter();
    let _ = iterator.filter_map(std::convert::identity);

    use std::convert::identity;
    let iterator = vec![Some(1), None, Some(2)].into_iter();
    let _ = iterator.filter_map(identity);

    let iterator = vec![Some(1), None, Some(2)].into_iter();
    let _ = iterator.filter_map(|x| return x);
}
