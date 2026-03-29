// Check that tuple indices in struct exprs & pats don't have a shorthand.
// If they did it would be possible to bind and reference numeric identifiers which is undesirable.

struct Rgb(u8, u8, u8);

fn main() {
    // FIXME: Also report a diagnostic for the other two fields.
    let Rgb { 0, 1, 2 };
    //~^ ERROR expected identifier, found `0`

    let _ = Rgb { 0, 1, 2 };
    //~^ ERROR expected identifier, found `0`
    //~| ERROR expected identifier, found `1`
    //~| ERROR expected identifier, found `2`
}
