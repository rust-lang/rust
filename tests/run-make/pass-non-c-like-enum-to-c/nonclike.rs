#[repr(C, u8)]
pub enum TT {
    AA(u64, u64),
    BB,
}

#[repr(C, u8)]
pub enum T {
    A(u64),
    B,
}

extern "C" {
    pub fn t_add(a: T, b: T) -> u64;
    pub fn tt_add(a: TT, b: TT) -> u64;
}

fn main() {
    assert_eq!(33, unsafe { tt_add(TT::AA(1, 2), TT::AA(10, 20)) });
    assert_eq!(11, unsafe { t_add(T::A(1), T::A(10)) });
}
