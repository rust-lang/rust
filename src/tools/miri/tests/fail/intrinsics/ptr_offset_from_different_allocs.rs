fn main() {
    unsafe {
        (&1_u8 as *const u8).offset_from(&2_u8); //~ERROR: not both derived from the same allocation
    }
}
