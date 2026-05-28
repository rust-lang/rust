pub fn pr32379(mut data: u64, f1: bool, f2: bool) -> u64 {
    if f1 { data &= !2; }
    if f2 { data |= 2; }
    data
}
