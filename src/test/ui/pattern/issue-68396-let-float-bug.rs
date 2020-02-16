fn main() {
    let 1234567890123456789012345678901234567890e-340: f64 = 0.0;
    //~^ ERROR could not evaluate float literal (see issue #31407)

    fn param(1234567890123456789012345678901234567890e-340: f64) {}
    //~^ ERROR could not evaluate float literal (see issue #31407)
}
