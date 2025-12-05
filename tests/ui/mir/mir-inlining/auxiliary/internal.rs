//@ compile-flags: -Zalways-encode-mir

static S: usize = 42;

pub fn f() -> &'static usize {
    &S
}
