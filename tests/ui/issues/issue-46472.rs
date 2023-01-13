fn bar<'a>() -> &'a mut u32 {
    &mut 4
    //~^ ERROR cannot return reference to temporary value [E0515]
}

fn main() { }
