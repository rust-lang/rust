// build-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn works() {
    let array/*: [u8; _]*/ = default_byte_array();
    let _: [_; 4] = array;
    Foo::foo(&array);
}

fn didnt_work() {
    let array/*: [u8; _]*/ = default_byte_array();
    Foo::foo(&array);
    let _: [_; 4] = array;
}

trait Foo<T> {
    fn foo(&self) {}
}

impl Foo<i32> for [u8; 4] {}
impl Foo<i64> for [u8; 8] {}

// Only needed because `[u8; _]` is not valid type syntax.
fn default_byte_array<const N: usize>() -> [u8; N]
where
    [u8; N]: Default,
{
    Default::default()
}

fn main() {
    works();
    didnt_work();
}
