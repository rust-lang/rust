mod _common;

use _common::validate;

fn main() {
    for i in 0..(1 << 19) {
        validate(&i.to_string());
    }
}
