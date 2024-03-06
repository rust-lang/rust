//@ check-pass

mod diagnostic {}

macro_rules! diagnostic{
    () => {}
}

#[allow(non_upper_case_globals)]
const diagnostic: () = ();

fn main() {
}
