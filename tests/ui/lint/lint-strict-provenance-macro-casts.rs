#![feature(strict_provenance_lints)]
#![deny(lossy_provenance_casts)]
#![deny(fuzzy_provenance_casts)]

macro_rules! cast {
    ($e:expr, $t:ty) => {
        $e as $t
        //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `usize`
        //~| ERROR strict provenance disallows casting integer `usize` to pointer `*const u8`
    };
}

macro_rules! p2i {
    ($e:expr) => { $e as usize };
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `usize`
}

macro_rules! i2p {
    ($e:expr) => { $e as *const () };
    //~^ ERROR strict provenance disallows casting integer `usize` to pointer `*const ()`
}

fn main() {
    let ptr = &0u8 as *const u8;
    let _addr = cast!(ptr, usize);

    let _ptr = cast!(0usize, *const u8);

    let x = 1u8;
    p2i!(&raw const x);

    i2p!(0x42);
}
