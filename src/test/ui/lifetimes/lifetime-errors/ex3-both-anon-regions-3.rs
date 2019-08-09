fn foo(z: &mut Vec<(&u8,&u8)>, (x, y): (&u8, &u8)) {
    z.push((x,y)); //~ ERROR lifetime mismatch
    //~^ ERROR lifetime mismatch
}

fn main() { }
