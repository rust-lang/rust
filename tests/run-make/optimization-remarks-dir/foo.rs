#[inline(never)]
pub fn bar() {}

#[inline(never)]
pub fn foo() {
    bar();
}
