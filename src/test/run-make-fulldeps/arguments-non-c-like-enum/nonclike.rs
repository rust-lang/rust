#[repr(C, u8)]
pub enum TT {
    AA(u64, u64),
    BB,
}

#[no_mangle]
pub extern "C" fn tt_add(a: TT, b: TT) -> u64 {
    match (a, b) {
        (TT::AA(a1, b1), TT::AA(a2, b2)) => a1 + a2 + b1 + b2,
        (TT::AA(a1, b1), TT::BB) => a1 + b1,
        (TT::BB, TT::AA(a1, b1)) => a1 + b1,
        _ => 0,
    }
}

#[repr(C, u8)]
pub enum T {
    A(u64),
    B,
}

#[no_mangle]
pub extern "C" fn t_add(a: T, b: T) -> u64 {
    match (a, b) {
        (T::A(a), T::A(b)) => a + b,
        (T::A(a), T::B) => a,
        (T::B, T::A(b)) => b,
        _ => 0,
    }
}
