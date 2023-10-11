fn main() {
    format_args!("{}\n", 0xffff_ffff_u8); //~ ERROR literal out of range for `u8`
}
