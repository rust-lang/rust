// Compiler:
//
// Run-time:
//   status: 0

use std::hint::black_box;
use std::num::NonZero;

fn main() {
    for (dividend, divisor, expected) in
        [(10u8, 3u8, 4u8), (1, 254, 1), (1, 255, 1), (2, 254, 1), (2, 255, 1), (200, 100, 2)]
    {
        let dividend = NonZero::new(black_box(dividend)).unwrap();
        let divisor = NonZero::new(black_box(divisor)).unwrap();
        assert_eq!(dividend.div_ceil(divisor).get(), expected);
    }
}
