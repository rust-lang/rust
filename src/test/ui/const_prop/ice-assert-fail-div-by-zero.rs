// check-pass

// need to emit MIR, because const prop (which emits `unconditional_panic`) only runs if
// the `optimized_mir` query is run, which it isn't in check-only mode.
// compile-flags: --crate-type lib --emit=mir,link

#![warn(unconditional_panic)]

pub struct Fixed64(i64);

// HACK: this test passes only because this is a const fn that is written to metadata
pub const fn div(f: Fixed64) {
    f.0 / 0; //~ WARN will panic at runtime
}
