fn main() {
    let x = [@{mutable a: @10, b: @20}];
    for @{a, b} in x {
        assert *a == 10;
        (*x[0]).a = @30;
        assert *a == 10;
    }
}
