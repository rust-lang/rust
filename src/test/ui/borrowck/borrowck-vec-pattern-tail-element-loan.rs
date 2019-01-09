#![feature(slice_patterns)]

fn a<'a>() -> &'a isize {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR `vec` does not live long enough
    let tail = match vec {
        &[_a, ref tail..] => &tail[0],
        _ => panic!("foo")
    };
    tail
}

fn main() {
    let fifth = a();
    println!("{}", *fifth);
}
