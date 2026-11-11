fn part(_: u16) -> u32 {
    1
}

fn main() {
    for n in 100_000.. {
    //~^ ERROR: literal out of range for `u16`
        let _ = part(n);
    }
}
