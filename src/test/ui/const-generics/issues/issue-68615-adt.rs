// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

struct Const<const V: [usize; 0]> {}
type MyConst = Const<{ [] }>;

fn main() {
    let _x = Const::<{ [] }> {};
    let _y = MyConst {};
}
