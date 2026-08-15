//@ run-pass
#![allow(unused_variables)]


macro_rules! vec [
    ($($e:expr),*) => ({
        let mut _temp = ::std::vec::Vec::new();
        $(_temp.push($e);)*
        _temp
    })
];

pub fn main() {
    let my_vec = vec![1, 2, 3, 4, 5];
}
