fn main() {
    let f = fn@() -> int {
        let i: int;
        ret i; //! ERROR use of possibly uninitialized variable: `i`
    };
    log(error, f());
}
