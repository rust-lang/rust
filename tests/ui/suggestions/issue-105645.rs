fn main() {
    let mut buf = [0u8; 50];
    let mut bref = buf.as_slice();
    foo(&mut bref);
    //~^ ERROR the trait bound `&[u8]: std::io::Write` is not satisfied [E0277]
}

fn foo(_: &mut impl std::io::Write) {}
