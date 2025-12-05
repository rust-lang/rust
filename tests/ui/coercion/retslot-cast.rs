#![allow(warnings)]

pub fn fail(x: Option<&(dyn Iterator<Item=()>+Send)>)
            -> Option<&dyn Iterator<Item=()>> {
    // This call used to trigger an LLVM assertion because the return
    // slot had type "Option<&Iterator>"* instead of
    // "Option<&(Iterator+Send)>"* -- but this now yields a
    // compilation error and I'm not sure how to create a comparable
    // test. To ensure that this PARTICULAR failure doesn't occur
    // again, though, I've left this test here, so if this ever starts
    // to compile again, we can adjust the test appropriately (clearly
    // it should never ICE...). -nmatsakis
    inner(x) //~ ERROR mismatched types
}

pub fn inner(x: Option<&(dyn Iterator<Item=()>+Send)>)
             -> Option<&(dyn Iterator<Item=()>+Send)> {
    x
}


fn main() {}
