use std::{fs::File, io::BufReader};

struct F(BufReader<File>);

fn main() {
    let f = F(BufReader::new(File::open("x").unwrap()));
    let x = f.get_ref(); //~ ERROR E0599
    //~^ HELP one of the expressions' fields has a method of the same name
    //~| HELP consider pinning the expression
    let f = (BufReader::new(File::open("x").unwrap()), );
    let x = f.get_ref(); //~ ERROR E0599
    //~^ HELP one of the expressions' fields has a method of the same name
    //~| HELP consider pinning the expression

    // FIXME(estebank): the pinning suggestion should not be included in either case.
    // https://github.com/rust-lang/rust/issues/144602
}
