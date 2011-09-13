// error-pattern:invalidate reference x

fn main() {
    let v: [mutable {mutable x: int}] = [mutable {mutable x: 1}];
    for x in v { v[0] = {mutable x: 2}; log x; }
}
