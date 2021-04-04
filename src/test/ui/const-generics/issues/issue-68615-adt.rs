// [full] check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Const<const V: [usize; 0]> {}
//[min]~^ ERROR `[usize; 0]` is forbidden as the type of a const generic parameter
type MyConst = Const<{ [] }>;

fn main() {
    let _x = Const::<{ [] }> {};
    let _y = MyConst {};
}
