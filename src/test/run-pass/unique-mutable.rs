fn main() {
    let i = ~mut 0;
    *i = 1;
    assert *i == 1;
}