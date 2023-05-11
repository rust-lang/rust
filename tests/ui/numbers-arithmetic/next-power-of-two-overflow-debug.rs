// run-pass
// compile-flags: -C debug_assertions=true
// needs-unwind
// ignore-emscripten dies with an LLVM error

use std::panic;

fn main() {
    macro_rules! overflow_test {
        ($t:ident) => (
            let r = panic::catch_unwind(|| {
                ($t::MAX).next_power_of_two()
            });
            assert!(r.is_err());

            let r = panic::catch_unwind(|| {
                (($t::MAX >> 1) + 2).next_power_of_two()
            });
            assert!(r.is_err());
        )
    }
    overflow_test!(u8);
    overflow_test!(u16);
    overflow_test!(u32);
    overflow_test!(u64);
    overflow_test!(u128);
}
