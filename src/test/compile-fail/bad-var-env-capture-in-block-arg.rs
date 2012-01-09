fn main() {
    let x = 3;
    fn blah(_a: fn()) {}
    blah({||
        log(debug, x); //! ERROR attempted dynamic environment capture
    });
}