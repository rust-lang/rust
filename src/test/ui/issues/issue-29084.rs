macro_rules! foo {
    ($d:expr) => {{
        fn bar(d: u8) { }
        bar(&mut $d);
        //~^ ERROR mismatched types
        //~| expected u8, found &mut u8
        //~| expected type `u8`
        //~| found mutable reference `&mut u8`
    }}
}

fn main() {
    foo!(0u8);
    //~^ in this expansion of foo!
}
