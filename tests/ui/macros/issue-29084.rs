macro_rules! foo {
    ($d:expr) => {{
        fn bar(d: u8) { }
        bar(&mut $d);
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected `u8`, found `&mut u8`
    }}
}

fn main() {
    foo!(0u8);
    //~^ NOTE_NONVIRAL in this expansion of foo!
}
