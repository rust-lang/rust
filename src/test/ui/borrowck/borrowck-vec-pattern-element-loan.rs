#![feature(slice_patterns)]

fn a<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let tail = match vec {
        &[_, ref tail..] => tail,
        _ => panic!("a")
    };
    tail
}

fn b<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let init = match vec {
        &[ref init.., _] => init,
        _ => panic!("b")
    };
    init
}

fn c<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let slice = match vec {
        &[_, ref slice.., _] => slice,
        _ => panic!("c")
    };
    slice
}

fn main() {}
