// This test checks that we lint on Option of fn ptr.
//
// https://github.com/rust-lang/rust/issues/134527.
//
//@ check-pass

unsafe extern "C" fn func() {}

type FnPtr = unsafe extern "C" fn();

fn main() {
    let _ = Some::<FnPtr>(func) == Some(func as unsafe extern "C" fn());
    //~^ WARN function pointer comparisons

    assert_eq!(Some::<FnPtr>(func), Some(func as unsafe extern "C" fn()));
    //~^ WARN function pointer comparisons
}
