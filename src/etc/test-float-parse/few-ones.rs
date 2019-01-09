mod _common;

use _common::validate;

fn main() {
    let mut pow = vec![];
    for i in 0..63 {
        pow.push(1u64 << i);
    }
    for a in &pow {
        for b in &pow {
            for c in &pow {
                validate(&(a | b | c).to_string());
            }
        }
    }
}
