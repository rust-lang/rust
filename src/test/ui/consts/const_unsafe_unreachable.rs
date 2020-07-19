// run-pass

#![feature(const_fn)]
#![feature(const_unreachable_unchecked)]

const unsafe fn foo(x: bool) -> bool {
    match x {
        true => true,
        false => std::hint::unreachable_unchecked(),
    }
}

const BAR: bool = unsafe { foo(true) };

fn main() {
  assert_eq!(BAR, true);
}
