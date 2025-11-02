//@ compile-flags: --test

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
mod test {}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
mod loooooooooooooong_teeeeeeeeeest {
    /*
    this is a comment
    this comment goes on for a very long time
    this is to pad out the span for this module for a long time
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
    labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
    laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
    voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
    non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    */
}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
extern "C" {}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
trait Foo {}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
impl Foo for i32 {}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
const FOO: i32 = -1_i32;

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
static BAR: u64 = 10_000_u64;

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
enum MyUnit {
    Unit,
}

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
struct NewI32(i32);

#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
union Spooky {
    x: i32,
    y: u32,
}

#[repr(C, align(64))]
#[test] //~ ERROR: the `#[test]` attribute may only be used on a free function
#[derive(Copy, Clone, Debug)]
struct MoreAttrs {
    a: i32,
    b: u64,
}

macro_rules! foo {
    () => {};
}

#[test] //~ WARN: the `#[test]` attribute may only be used on a free function
foo!();

// make sure it doesn't erroneously trigger on a real test
#[test]
fn real_test() {
    assert_eq!(42_i32, 42_i32);
}

// make sure it works with cfg test
#[cfg(test)]
mod real_tests {
    #[cfg(test)]
    fn foo() {}

    #[test]
    fn bar() {
        foo();
    }
}
