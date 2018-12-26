mod _common;

use _common::validate;

fn main() {
    // Skip e = 0 because small-u32 already does those.
    for e in 1..301 {
        for i in 0..10000 {
            // If it ends in zeros, the parser will strip those (and adjust the exponent),
            // which almost always (except for exponents near +/- 300) result in an input
            // equivalent to something we already generate in a different way.
            if i % 10 == 0 {
                continue;
            }
            validate(&format!("{}e{}", i, e));
            validate(&format!("{}e-{}", i, e));
        }
    }
}
