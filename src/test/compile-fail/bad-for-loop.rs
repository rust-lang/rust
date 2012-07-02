fn main() {
    fn baz(_x: fn() -> int) {}
    for baz |_e| { } //~ ERROR should return `bool`
}
