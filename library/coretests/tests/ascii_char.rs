use core::ascii::Char;
use core::fmt::Write;

/// Tests Display implementation for ascii::Char.
#[test]
fn test_display() {
    let want = (0..128u8).map(|b| b as char).collect::<String>();
    let mut got = String::with_capacity(128);
    for byte in 0..128 {
        write!(&mut got, "{}", Char::from_u8(byte).unwrap()).unwrap();
    }
    assert_eq!(want, got);
}

/// Tests Debug implementation for ascii::Char.
#[test]
fn test_debug_control() {
    for byte in 0..128u8 {
        let mut want = format!("{:?}", byte as char);
        // `char` uses `'\u{#}'` representation where ascii::char uses `'\x##'`.
        // Transform former into the latter.
        if let Some(rest) = want.strip_prefix("'\\u{") {
            want = format!("'\\x{:0>2}'", rest.strip_suffix("}'").unwrap());
        }
        let chr = core::ascii::Char::from_u8(byte).unwrap();
        assert_eq!(want, format!("{chr:?}"), "byte: {byte}");
    }
}

/// Tests Extend implementation for ascii::Char.
#[test]
fn test_extend() {
    let mut s = String::from("abc");
    s.extend_one(Char::SmallD);
    assert_eq!(s, String::from("abcd"));

    let mut s = String::from("abc");
    s.extend(Char::CapitalA..=Char::CapitalC);
    assert_eq!(s, String::from("abcABC"));
}

#[test]
fn test_range_inclusive_try_fold() {
    // Visit every value once.
    let mut it = Char::MIN..=Char::MAX;
    let mut expected: u32 = 0;
    it.try_fold((), |_, x| {
        assert!(expected <= u32::from(Char::MAX));
        assert_eq!(expected, u32::from(x));
        expected += 1;
        Some(())
    });

    let mut it = Char::MIN..=Char::MAX;
    it.try_fold((), |_, x| (x < Char::CapitalA).then_some(()));
    assert!(!it.is_empty());
    assert_eq!(it.next(), Some(Char::CapitalB));

    let mut it = Char::MIN..=Char::MAX;
    it.try_fold((), |_, x| (x != Char::MAX).then_some(()));
    assert!(it.is_empty());
    assert_eq!(it.next(), None);

    // Visit every value once.
    let mut it = Char::MIN..=Char::MAX;
    let mut expected: i32 = u8::from(Char::MAX).into();
    it.try_rfold((), |_, x| {
        assert!(expected >= 0);
        assert_eq!(expected, i32::from(u8::from(x)));
        expected -= 1;
        Some(())
    });

    let mut it = Char::MIN..=Char::MAX;
    it.try_rfold((), |_, x| (x > Char::CapitalB).then_some(()));
    assert!(!it.is_empty());
    assert_eq!(it.next_back(), Some(Char::CapitalA));

    let mut it = Char::MIN..=Char::MAX;
    it.try_rfold((), |_, x| (x != Char::MIN).then_some(()));
    assert!(it.is_empty());
    assert_eq!(it.next(), None);
}
