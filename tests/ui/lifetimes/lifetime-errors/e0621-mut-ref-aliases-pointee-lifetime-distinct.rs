// Regression test for #156682
// When the suggested type would NOT alias the outer reference's lifetime with one
// already used inside the pointee, the normal suggestion should still apply.

pub fn push<'a>(x: &mut Vec<&'a u8>, y: &u8) {
    x.push(y);
    //~^ ERROR explicit lifetime required in the type of `y`
}

fn main() {}
