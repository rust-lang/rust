fn main() {
    unknown(1, |glyf| {
        //~^ ERROR: cannot find function `unknown`
        let actual = glyf;
    });
}
