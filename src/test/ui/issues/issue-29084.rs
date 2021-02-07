macro_rules! foo {
    ($d:expr) => {{
        fn bar(d: u8) { }
        bar(&mut $d);
        //~^ ERROR arguments to this function are incorrect
        //~| expected `u8`, found `&mut u8`
    }}
}

fn main() {
    foo!(0u8);
    //~^ in this expansion of foo!
}
