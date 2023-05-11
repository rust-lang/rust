fn foo(x: &mut Vec<&u8>, y: &u8) {
    x.push(y);
    //~^ ERROR lifetime may not live long enough
}

fn main() { }
