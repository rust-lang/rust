fn foo(x: &u32) -> &'static u32 {
    &*x
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
