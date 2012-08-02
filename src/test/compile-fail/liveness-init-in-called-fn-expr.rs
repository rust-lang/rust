fn main() {
    let j = fn@() -> int {
        let i: int;
        return i; //~ ERROR use of possibly uninitialized variable: `i`
    };
    j();
}
