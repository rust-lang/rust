type BOOL = i32;

#[cfg_attr(
    target_arch = "x86",
    link(name = "bcryptprimitives", kind = "raw-dylib", import_name_type = "undecorated")
)]
#[cfg_attr(not(target_arch = "x86"), link(name = "bcryptprimitives", kind = "raw-dylib"))]
extern "system" {
    fn ProcessPrng(pbdata: *mut u8, cbdata: usize) -> BOOL;
}

fn main() {
    let mut num: u8 = 0;
    unsafe {
        ProcessPrng(&mut num, 1);
    }
    println!("{num}");
}
