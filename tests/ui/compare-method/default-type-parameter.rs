trait Trait<T = ()> {
    fn foo() -> T;
}

struct S;

impl Trait for S {
    fn foo() -> u32 { 0 } //~ ERROR
}
impl Trait<i32> for () {
    fn foo() -> u32 { 0 } //~ ERROR
}

fn main() {}
