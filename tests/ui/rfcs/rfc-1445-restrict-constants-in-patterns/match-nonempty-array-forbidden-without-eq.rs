// Issue 62307 pointed out a case where the structural-match checking
// was too shallow.
//
// Here we check similar behavior for non-empty arrays of types that
// do not derive `Eq`.
//
// (Current behavior for empty arrays differs and thus is not tested
// here; see rust-lang/rust#62336.)

#[derive(PartialEq, Debug)]
struct B(i32);

fn main() {
    const FOO: [B; 1] = [B(0)];
    match [B(1)] {
        FOO => { }
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
    }
}
