#[inline(always)] //~ ERROR: E0518
struct Foo;

#[inline(never)] //~ ERROR: E0518
impl Foo {
}

fn main() {
}
