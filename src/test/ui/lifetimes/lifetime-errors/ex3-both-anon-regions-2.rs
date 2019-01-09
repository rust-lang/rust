fn foo(&mut (ref mut v, w): &mut (&u8, &u8), x: &u8) {
    *v = x; //~ ERROR lifetime mismatch
}

fn main() { }
