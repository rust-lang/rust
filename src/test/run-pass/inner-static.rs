// aux-build:inner_static.rs


extern crate inner_static;

pub fn main() {
    let a = inner_static::A::<()> { v: () };
    let b = inner_static::B::<()> { v: () };
    let c = inner_static::test::A::<()> { v: () };
    assert_eq!(a.bar(), 2);
    assert_eq!(b.bar(), 4);
    assert_eq!(c.bar(), 6);
}
