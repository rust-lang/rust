fn main() {
    format_args!("{}", 0x8f_i8); // issue #115423
    //~^ ERROR literal out of range for `i8`
    format_args!("{}", 0xffff_ffff_u8); // issue #116633
    //~^ ERROR literal out of range for `u8`
    format_args!("{}", 0xffff_ffff); // treat unsuffixed literals as i32
    //~^ ERROR literal out of range for `i32`
}
