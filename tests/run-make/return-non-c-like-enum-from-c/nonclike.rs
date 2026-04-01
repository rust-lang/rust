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
    pub fn t_new(a: u64) -> T;
    pub fn tt_new(a: u64, b: u64) -> TT;
}

fn main() {
    if let TT::AA(a, b) = unsafe { tt_new(10, 11) } {
        assert_eq!(10, a);
        assert_eq!(11, b);
    } else {
        panic!("expected TT::AA");
    }

    if let T::A(a) = unsafe { t_new(10) } {
        assert_eq!(10, a);
    } else {
        panic!("expected T::A");
    }
}
