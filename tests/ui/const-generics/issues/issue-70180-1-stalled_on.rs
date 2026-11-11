//@ build-pass

pub fn works() {
    let array/*: [_; _]*/ = default_array();
    let _: [_; 4] = array;
    Foo::foo(&array);
}

pub fn didnt_work() {
    let array/*: [_; _]*/ = default_array();
    Foo::foo(&array);
    let _: [_; 4] = array;
}

trait Foo {
    fn foo(&self) {}
}

impl Foo for [i32; 4] {}
impl Foo for [i64; 8] {}

// Only needed because `[_; _]` is not valid type syntax.
fn default_array<T, const N: usize>() -> [T; N]
where
    [T; N]: Default,
{
    Default::default()
}

fn main() {
    works();
    didnt_work();
}
