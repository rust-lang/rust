#![feature(strict_provenance)]
#![deny(lossy_provenance_casts)]

fn main() {
    let x: u8 = 37;
    let addr: usize = &x as *const u8 as usize;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `usize`

    let addr_32bit = &x as *const u8 as u32;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `u32`
}
