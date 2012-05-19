fn main() {
    let j = fn@() -> int {
        let i: int;
        ret i; //! ERROR use of possibly uninitialized variable: `i`
    };
    j();
}
