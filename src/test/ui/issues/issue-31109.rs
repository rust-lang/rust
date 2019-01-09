fn main() {
    // FIXME(#31407) this error should go away, but in the meantime we test that it
    // is accompanied by a somewhat useful error message.
    let _: f64 = 1234567890123456789012345678901234567890e-340;
    //~^ ERROR could not evaluate float literal (see issue #31407)
}
