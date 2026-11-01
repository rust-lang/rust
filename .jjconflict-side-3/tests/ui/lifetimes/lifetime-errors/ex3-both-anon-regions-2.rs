fn foo(&mut (ref mut v, w): &mut (&u8, &u8), x: &u8) {
    //~^ ERROR lifetime may not live long enough
    *v = x;
}

fn main() { }
