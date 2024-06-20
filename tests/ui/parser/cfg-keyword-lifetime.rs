// Disallow `'keyword` even in cfg'd code.

#[cfg(any())]
fn hello() -> &'ref () {}
//~^ ERROR lifetimes cannot use keyword names

macro_rules! macro_invocation {
    ($i:item) => {}
}
macro_invocation! {
    fn hello() -> &'ref () {}
    //~^ ERROR lifetimes cannot use keyword names
}

fn main() {}
