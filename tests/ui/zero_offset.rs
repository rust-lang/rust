fn main() {
    unsafe {
        let x = &() as *const ();
        x.offset(0);
        x.wrapping_add(0);
        x.sub(0);
        x.wrapping_sub(0);

        let y = &1 as *const u8;
        y.offset(0);
    }
}
