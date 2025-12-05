#![feature(strict_provenance_lints)]
#![deny(lossy_provenance_casts)]

fn main() {
    let x: u8 = 37;
    let addr: usize = &x as *const u8 as usize;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `usize`

    let addr_32bit = &x as *const u8 as u32;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `u32`

    // don't add unnecessary parens in the suggestion
    let ptr = &x as *const u8;
    let ptr_addr = ptr as usize;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `usize`
    let ptr_addr_32bit = ptr as u32;
    //~^ ERROR under strict provenance it is considered bad style to cast pointer `*const u8` to integer `u32`
}
