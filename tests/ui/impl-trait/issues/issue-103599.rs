trait T {}

fn wrap(x: impl T) -> impl T {
    //~^ ERROR cannot resolve opaque type
    //~| WARN function cannot return without recursing
    wrap(wrap(x))
}

fn main() {}
