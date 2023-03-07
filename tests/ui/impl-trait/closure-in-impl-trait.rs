// run-pass
#![allow(unused_must_use)]
fn bug<T>() -> impl Iterator<Item = [(); { |x: u32| { x }; 4 }]> {
    std::iter::empty()
}

fn ok<T>() -> Box<dyn Iterator<Item = [(); { |x: u32| { x }; 4 }]>> {
    Box::new(std::iter::empty())
}

fn main() {
    for _item in ok::<u32>() {}
    for _item in bug::<u32>() {}
}
