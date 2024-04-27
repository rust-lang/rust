fn foo(z: &mut Vec<(&u8,&u8)>, (x, y): (&u8, &u8)) {
    z.push((x,y));
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn main() { }
