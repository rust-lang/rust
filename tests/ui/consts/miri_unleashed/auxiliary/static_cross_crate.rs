pub static mut ZERO: [u8; 1] = [0];
pub static ZERO_REF: &[u8; 1] = unsafe { &ZERO };
pub static mut OPT_ZERO: Option<u8> = Some(0);
