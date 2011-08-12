// error-pattern:invalidate alias x

fn main() {
    let v: [mutable int] = ~[mutable 1, 2, 3];
    for x: int  in v { v.(0) = 10; log x; }
}