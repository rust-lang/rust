#[no_mangle]
pub static mut MARK: u8 = 0;

#[no_mangle]
pub extern "C" fn mark_ctor() {
    unsafe {
        MARK = 1;
    }
    eprintln!("constructor ran");
}

#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn() = mark_ctor;
