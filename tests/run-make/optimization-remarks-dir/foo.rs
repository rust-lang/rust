#[inline(never)]
pub fn bar() {}

pub fn foo() {
    bar();
}
