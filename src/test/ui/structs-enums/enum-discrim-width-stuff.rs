// run-pass
#![allow(overflowing_literals)]
#![allow(dead_code)]

macro_rules! check {
    ($m:ident, $t:ty, $v:expr) => {{
        mod $m {
            use std::mem::size_of;
            #[derive(Copy, Clone, Debug)]
            enum E {
                V = $v,
                A = 0
            }
            static C: E = E::V;
            pub fn check() {
                assert_eq!(size_of::<E>(), size_of::<$t>());
                assert_eq!(E::V as $t, $v as $t);
                assert_eq!(C as $t, $v as $t);
                assert_eq!(format!("{:?}", E::V), "V".to_string());
                assert_eq!(format!("{:?}", C), "V".to_string());
            }
        }
        $m::check();
    }}
}

pub fn main() {
    check!(a, u8, 0x17);
    check!(b, u8, 0xe8);
    check!(c, u16, 0x1727);
    check!(d, u16, 0xe8d8);
    check!(e, u32, 0x17273747);
    check!(f, u32, 0xe8d8c8b8);

    check!(z, i8, 0x17);
    check!(y, i8, -0x17);
    check!(x, i16, 0x1727);
    check!(w, i16, -0x1727);
    check!(v, i32, 0x17273747);
    check!(u, i32, -0x17273747);

    enum Simple { A, B }
    assert_eq!(::std::mem::size_of::<Simple>(), 1);
}
