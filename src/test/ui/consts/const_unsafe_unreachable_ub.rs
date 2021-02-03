// build-fail

#![feature(const_fn)]
#![feature(const_unreachable_unchecked)]

const unsafe fn foo(x: bool) -> bool {
    match x {
        true => true,
        false => std::hint::unreachable_unchecked(),
    }
}

#[warn(const_err)]
const BAR: bool = unsafe { foo(false) };

fn main() {
  assert_eq!(BAR, true);
  //~^ ERROR E0080
  //~| ERROR erroneous constant
  //~| WARN this was previously accepted by the compiler but is being phased out
}
