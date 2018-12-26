// Make sure that inclusive ranges with no end point don't parse.

// compile-flags: -Z parse-only

pub fn main() {
    for _ in 1..= {} //~ERROR inclusive range with no end
                     //~^HELP bounded at the end
}
