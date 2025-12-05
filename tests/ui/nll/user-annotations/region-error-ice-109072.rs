// Regression test for #109072.
// Check that we don't ICE when canonicalizing user annotation.

trait Lt<'a> {
    type T;
}

impl Lt<'missing> for () { //~ ERROR undeclared lifetime
    type T = &'missing (); //~ ERROR undeclared lifetime
}

fn main() {
    let _: <() as Lt<'_>>::T = &();
}
