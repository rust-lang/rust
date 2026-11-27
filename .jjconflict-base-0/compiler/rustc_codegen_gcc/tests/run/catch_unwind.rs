// Compiler:
//
// Run-time:
//   status: 0
//   stdout: Caught

#![feature(fn_traits, unboxed_closures)]

struct Wrapper<A>(A);

impl<R, F: FnOnce() -> R> FnOnce<()> for Wrapper<F> {
    type Output = R;

    #[inline]
    extern "rust-call" fn call_once(self, _args: ()) -> R {
        (self.0)()
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(Wrapper(|| panic!()));
    assert!(result.is_err());
    println!("Caught");
}
