fn main() {
    let i = ~mutable 0;
    *i = 1;
    assert *i == 1;
}