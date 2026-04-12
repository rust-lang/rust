// Syntactically, tuple indices have to be of the form `0|[1-9][0-9]*`.
// What makes this slightly annoying for the parser is the fact that lexically we only have
// int and float literals which adhere to a larger grammar. For that reason, the parser has
// to manully introspect these tokens.

// See also <https://github.com/rust-lang/rust/issues/47073>.

fn scope() {
    // Note that we normalize these faulty indices "as you expected" (i.e., dropping underscores,
    // leading zeroes, non-digits[^1]) to suppress bad follow-up errors like "unknown field" and
    // to force good ones like "private field".
    //
    // [^1]: FIXME(fmease): We shouldn't blindly drop alphabetics and instead take into account the
    //                      "requested" base & exponent.

    // Subsequently we interpret `00` as `0`.
    let _ = (0,).00; //~ ERROR invalid tuple index

    // Subsequently we interpret `01_00` as `100`.
    let _ = ().01_00; //~ ERROR invalid tuple index
    //~^ ERROR no field `100` on type `()`

    struct Xyz(f32, f32, f32);

    // Subsequently we interpret `02` as `2`.
    let Xyz { 02: _, .. }; //~ ERROR invalid tuple index

    // Subsequently we interpret `3_000` as `3000`.
    let Xyz { 3_000: _, .. }; //~ ERROR invalid tuple index
    //~^ ERROR struct `Xyz` does not have a field named `3000`

    // Subsequently we interpret `0x0` as `0`, `01` as `1` and `2__` as `2`.
    let _ = Xyz { 0x0: 0., 01: 0., 2__: 0. };
    //~^ ERROR invalid tuple index
    //~| ERROR invalid tuple index
    //~| ERROR invalid tuple index
}

// This is cfg'ed out to ensure that these checks are syntactic not just semantic.
#[cfg(false)]
fn scope() {
    let X { 00: _, 1_0: _, 0xF: _ };
    //~^ ERROR invalid tuple index
    //~| ERROR invalid tuple index
    //~| ERROR invalid tuple index

    let _ = X { 001: (), 1_000: (), 0o7: () };
    //~^ ERROR invalid tuple index
    //~| ERROR invalid tuple index
    //~| ERROR invalid tuple index

    // FIXME(fmease): Also emit "invalid tuple index" here:
    let X { 1e1: _ } = X { 100e10: () };
    //~^ ERROR expected identifier, found `1e1`
    //~| ERROR expected identifier, found `100e10`

    let _ = x.0001; //~ ERROR invalid tuple index
    let _ = x.1_000; //~ ERROR invalid tuple index
    let _ = x.0b1010; //~ ERROR invalid tuple index
    let _ = x.2e3; //~ ERROR invalid tuple index
    let _ = x.1.00; //~ ERROR invalid tuple index
    let _ = x.1.1_; //~ ERROR invalid tuple index
    let _ = x.1.1e1; //~ ERROR invalid tuple index

    // FIXME(fmease): Also emit "invalid tuple index" here and maybe recover from it:
    let _ = x.1e+10;
    //~^ ERROR unexpected token: `1e+10`
    //~| ERROR expected one of
    // FIXME(fmease): Maybe recover from above and emit "invalid tuple index" below:
    let _ = x.1e-10;
}

fn main() {}
