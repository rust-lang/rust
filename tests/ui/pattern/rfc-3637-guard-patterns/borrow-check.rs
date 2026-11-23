// Check whether the borrow checker runs in guard patterns

#![feature(guard_patterns)]

struct NonCopy(u8);

pub fn probe(
    (_ if const {
    let x = NonCopy(0);
    let y = x;
    let z = x;
    //~^ ERROR use of moved value: `x` [E0382]
    true
}): u8,
) {}

fn main() {}
