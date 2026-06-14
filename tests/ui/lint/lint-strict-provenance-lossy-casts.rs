#![feature(strict_provenance_lints)]
#![deny(implicit_provenance_casts)]

fn main() {
    let x: u8 = 37;
    let addr: usize = &x as *const u8 as usize;
    //~^ ERROR cast from `*const u8` to `usize` implicitly exposes pointer provenance

    let addr_32bit = &x as *const u8 as u32;
    //~^ ERROR cast from `*const u8` to `u32` implicitly exposes pointer provenance

    // don't add unnecessary parens in the suggestion
    let ptr = &x as *const u8;
    let ptr_addr = ptr as usize;
    //~^ ERROR cast from `*const u8` to `usize` implicitly exposes pointer provenance
    let ptr_addr_32bit = ptr as u32;
    //~^ ERROR cast from `*const u8` to `u32` implicitly exposes pointer provenance
}
