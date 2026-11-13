use crate::sys::pal::windows::api::{utf16, wide_str};

macro_rules! check_utf16 {
    ($str:literal) => {{
        assert!(wide_str!($str).iter().copied().eq($str.encode_utf16().chain([0])));
        assert!(utf16!($str).iter().copied().eq($str.encode_utf16()));
    }};
}

#[test]
fn test_utf16_macros() {
    check_utf16!("hello world");
    check_utf16!("€4.50");
    check_utf16!("𨉟呐㗂越");
    check_utf16!("Pchnąć w tę łódź jeża lub ośm skrzyń fig");
}
