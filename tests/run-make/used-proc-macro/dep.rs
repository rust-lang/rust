#![crate_type = "lib"]

#[used]
#[no_mangle] // so we can refer to this variable by name from the proc-macro library
static VERY_IMPORTANT_SYMBOL: u32 = 12345;
