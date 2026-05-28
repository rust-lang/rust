// https://github.com/rust-lang/rustfmt/issues/3623

fn foo(#[cfg(something)] x: i32, y: i32) -> i32 {
    x + y
}

fn foo_b(#[cfg(something)] x: i32, y: i32) -> i32 {
    x + y
}

fn add(
    #[cfg(something)]
    #[deny(C)]
    x: i32,
    y: i32,
) -> i32 {
    x + y
}

struct NamedSelfRefStruct {}
impl NamedSelfRefStruct {
    fn foo(#[cfg(something)] self: &Self) {}
}

struct MutStruct {}
impl MutStruct {
    fn foo(#[cfg(foo)] &mut self, #[deny(C)] b: i32) {}
}

fn main() {
    let c = |#[allow(C)] a: u32,
             #[cfg(something)] b: i32,
             #[cfg_attr(something, cfg(nothing))]
             #[deny(C)]
             c: i32| {};
    let _ = c(1, 2);
}

pub fn bar(
    /// bar
    #[test]
    a: u32,
    /// Bar
    #[must_use]
    /// Baz
    #[no_mangle]
    b: i32,
) {
}

fn abc(
    #[foo]
    #[bar]
    param: u32,
) {
    // ...
}

fn really_really_really_loooooooooooooooooooong(
    #[cfg(some_even_longer_config_feature_that_keeps_going_and_going_and_going_forever_and_ever_and_ever_on_and_on)]
    b: i32,
) {
    // ...
}
