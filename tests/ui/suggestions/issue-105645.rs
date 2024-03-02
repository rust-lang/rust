fn main() {
    let mut buf = [0u8; 50];
    let mut bref = buf.as_slice();
    foo(&mut bref);
    //~^ ERROR trait `std::io::Write` is not implemented for `&[u8]`
}

fn foo(_: &mut impl std::io::Write) {}
