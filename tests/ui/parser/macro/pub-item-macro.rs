// Issue #14660

macro_rules! priv_x {
    () => {
        static x: u32 = 0;
    };
}

macro_rules! pub_x { () => {
    pub priv_x!(); //~ ERROR can't qualify macro invocation with `pub`
    //~^ HELP remove the visibility
    //~| HELP try adjusting the macro to put `pub` inside the invocation
}}

mod foo {
    pub_x!();
}

fn main() {
    let y: u32 = foo::x; //~ ERROR static `x` is private
}
