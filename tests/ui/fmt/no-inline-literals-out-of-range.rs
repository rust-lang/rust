//@ only-64bit

fn main() {
    format_args!("{}", 0x8f_i8); // issue #115423
    //~^ ERROR literal out of range for `i8`
    format_args!("{}", 0xffff_ffff_u8); // issue #116633
    //~^ ERROR literal out of range for `u8`
    format_args!("{}", 0xffff_ffff_ffff_ffff_ffff_usize);
    //~^ ERROR literal out of range for `usize`
    format_args!("{}", 0x8000_0000_0000_0000_isize);
    //~^ ERROR literal out of range for `isize`
    format_args!("{}", 0xffff_ffff); // treat unsuffixed literals as i32
    //~^ ERROR literal out of range for `i32`
}
