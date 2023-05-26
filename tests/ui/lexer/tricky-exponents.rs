// This tests for integer literal suffixes involving 'e' that used to be mostly
// invalid, because they were seen as floats with invalid exponents, but were
// made valid for #111615.
//
// run-pass
// check-run-results

// Print and preserve the tokens. For tokens that the parser will accept.
//
// The `stringify!` call here is realistic: a proc macro that takes these
// kinds of tokens will typically just stringify and reparse the tokens itself.
// This means, for example, a proc macro taking CSS colours doesn't know or
// care that `aabbcc` is a an identifier token, while `112233` is an integer
// token, and `1122aa` is an integer token with a suffix.
macro_rules! p {
    ($x:tt) => { println!(stringify!($x)); _ = $x }
}

// Print and consume the tokens. For tokens that the parser will not accept.
macro_rules! q {
    ($($x:tt)*) => { println!(stringify!($($x)*)) }
}

fn main() {
    p!(1e_3);         // always: equivalent to `1e3` float
    p!(1e_3_);        // always: equivalent to `1e3` float
    p!(1e____3);      // always: equivalent to `1e3` float
    p!(1e____3_3__);  // always: equivalent to `1e33` float
    p!(1e+____3);     // always: equivalent to `1e+3` float
    p!(1e-____3_3__); // always: equivalent to `1e-33` float

    // Decimal integers with unit suffixes
    q!(1kg); // always: `1` int + `kg` suffix
    q!(1ns); // always: `1` int + `ns` suffix
    q!(1MW); // always: `1` int + `MW` suffix
    q!(1eV); // was: invalid, now: `1` int + `eV` suffix

    // Non-decimal integers with unit suffixes
    q!(0x1em); // always: `0x1e` int + `m` suffix (because `e` is a hex digit!)
    q!(0o1em); // was: invalid, now: `0o1` int + `em` suffix
    q!(0b1em); // was: invalid, now: `0b1` int + `em` suffix
    q!(0b1e ); // was: invalid, now: `0b1` int + `e` suffix

    // Non-exponent floats with unit suffixes
    q!(2.0px  ); // always: `2.0` + `px` suffix
    q!(2.0e   ); // was: invalid, now: `2.0` + `e` suffix
    q!(2.0em  ); // was: invalid, now: `2.0` + `em` suffix
    q!(2.0e3em); // always: `2.0e3` + `em` suffix

    // Decimal integers with suffixes containing underscores
    q!(0e_         ); // was: invalid, now `0` + `e_` suffix
    q!(1e_a        ); // was: invalid, now `1` + `e_a` suffix
    q!(1e_a_       ); // was: invalid, now `1` + `e_a_` suffix
    q!(1e++        ); // was: invalid, now `1` + `e` suffix + `+` ...
    q!(1e+___      ); // was: invalid, now `1` + `e` suffix + `+` ...
    q!(1e-_-       ); // was: invalid, now `1` + `e` suffix + `-` ...
    q!(1e+____a    ); // was: invalid, now `1` + `e` suffix + `+` ...
    q!(1e-____a_a__); // was: invalid, now `1` + `e` suffix + `-` ...

    // Floats with suffixes containing underscores
    q!(3.3e_____); // was: invalid: now `3.3` + `e_____` suffix

    // Exponent floats with unit suffixes
    q!(1e3foo);   // always: `1e3` + `foo` suffix
    q!(1.0e3foo); // always: `1.0e3` + `foo` suffix

    // CSS colours
    q!(#aabbcc); // always: `aabbcc` ident
    q!(#aabb11); // always: `aabb11` ident
    q!(#112233); // always: `112233` int
    q!(#1122aa); // always: `1122` int + `aa` suffix
    q!(#112e33); // always: `112e33` float
    q!(#112e3a); // always: `112e3` float + `a` suffix
    q!(#11223e); // was: invalid, now: `11223` + `e` suffix
    q!(#1122ea); // was: invalid, now: `1122` + `ea` suffix

    // UUIDs
    q!(7ad85a2c-f2d0-11fd-afd0-b3104db0cb68); // always: valid
    q!(7ad85a2c-f2d0-11ed-afd0-b3104db0cb68); // was: invalid (11ed), now ok
    q!(7ad85a2c-f2d0-111e-afd0-b3104db0cb68); // was: invalid (111e-a), now ok
}
