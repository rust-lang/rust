// Ensure that suggestions to search for missing intermediary field accesses are available for both
// tuple structs *and* regular tuples.
// Ensure that we do not suggest pinning the expression just because `Pin::get_ref` exists.
// https://github.com/rust-lang/rust/issues/144602
use std::{fs::File, io::BufReader};

struct F(BufReader<File>);

fn main() {
    let f = F(BufReader::new(File::open("x").unwrap()));
    let x = f.get_ref(); //~ ERROR E0599
    //~^ HELP one of the expressions' fields has a method of the same name
    let f = (BufReader::new(File::open("x").unwrap()), );
    let x = f.get_ref(); //~ ERROR E0599
    //~^ HELP one of the expressions' fields has a method of the same name
}
