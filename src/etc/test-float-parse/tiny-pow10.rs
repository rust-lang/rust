mod _common;

use _common::validate;

fn main() {
    for e in 301..327 {
        for i in 0..100000 {
            validate(&format!("{}e-{}", i, e));
        }
    }
}
