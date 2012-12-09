fn main() {
    fn baz(_x: fn(y: int) -> int) {}
    for baz |_e| { } //~ ERROR should return `bool`
}
