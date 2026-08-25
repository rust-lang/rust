fn main() {
    unknown(1, |glyf| {
        //~^ ERROR: cannot find function `unknown` in this scope
        let actual = glyf;
    });
}
