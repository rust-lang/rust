//@ dont-require-annotations: NOTE

macro_rules! foo {
    ($d:expr) => {{
        fn bar(d: u8) { }
        bar(&mut $d);
        //~^ ERROR mismatched types
        //~| NOTE expected `u8`, found `&mut u8`
    }}
}

fn main() {
    foo!(0u8);
    //~^ NOTE in this expansion of foo!
}
