// run-pass
#![allow(non_upper_case_globals)]

static mut destructions : isize = 3;

pub fn foo() {
    struct Foo;

    impl Drop for Foo {
        fn drop(&mut self) {
          unsafe { destructions -= 1 };
        }
    }

    let _x = [Foo, Foo, Foo];
}

pub fn main() {
  foo();
  assert_eq!(unsafe { destructions }, 0);
}
