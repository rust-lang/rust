#![allow(clippy::disallowed_types)]
use std::sync::Arc;

#[cfg(not(miri))]
use proptest::{prop_assert, prop_assert_eq, proptest};

use smol_str::{SmolStr, SmolStrBuilder};

#[test]
#[cfg(target_pointer_width = "64")]
fn smol_str_is_smol() {
    assert_eq!(::std::mem::size_of::<SmolStr>(), ::std::mem::size_of::<String>(),);
}

#[test]
fn assert_traits() {
    fn f<T: Send + Sync + ::std::fmt::Debug + Clone>() {}
    f::<SmolStr>();
}

#[test]
fn conversions() {
    let s: SmolStr = "Hello, World!".into();
    let s: String = s.into();
    assert_eq!(s, "Hello, World!");

    let s: SmolStr = Arc::<str>::from("Hello, World!").into();
    let s: Arc<str> = s.into();
    assert_eq!(s.as_ref(), "Hello, World!");
}

#[test]
fn const_fn_ctor() {
    const EMPTY: SmolStr = SmolStr::new_inline("");
    const A: SmolStr = SmolStr::new_inline("A");
    const HELLO: SmolStr = SmolStr::new_inline("HELLO");
    const LONG: SmolStr = SmolStr::new_inline("ABCDEFGHIZKLMNOPQRSTUVW");

    assert_eq!(EMPTY, SmolStr::from(""));
    assert_eq!(A, SmolStr::from("A"));
    assert_eq!(HELLO, SmolStr::from("HELLO"));
    assert_eq!(LONG, SmolStr::from("ABCDEFGHIZKLMNOPQRSTUVW"));
}

#[cfg(not(miri))]
fn check_props(std_str: &str, smol: SmolStr) -> Result<(), proptest::test_runner::TestCaseError> {
    prop_assert_eq!(smol.as_str(), std_str);
    prop_assert_eq!(smol.len(), std_str.len());
    prop_assert_eq!(smol.is_empty(), std_str.is_empty());
    if smol.len() <= 23 {
        prop_assert!(!smol.is_heap_allocated());
    }
    Ok(())
}

#[cfg(not(miri))]
proptest! {
    #[test]
    fn roundtrip(s: String) {
        check_props(s.as_str(), SmolStr::new(s.clone()))?;
    }

    #[test]
    fn roundtrip_spaces(s in r"( )*") {
        check_props(s.as_str(), SmolStr::new(s.clone()))?;
    }

    #[test]
    fn roundtrip_newlines(s in r"\n*") {
        check_props(s.as_str(), SmolStr::new(s.clone()))?;
    }

    #[test]
    fn roundtrip_ws(s in r"( |\n)*") {
        check_props(s.as_str(), SmolStr::new(s.clone()))?;
    }

    #[test]
    fn from_string_iter(slices in proptest::collection::vec(".*", 1..100)) {
        let string: String = slices.iter().map(|x| x.as_str()).collect();
        let smol: SmolStr = slices.into_iter().collect();
        check_props(string.as_str(), smol)?;
    }

    #[test]
    fn from_str_iter(slices in proptest::collection::vec(".*", 1..100)) {
        let string: String = slices.iter().map(|x| x.as_str()).collect();
        let smol: SmolStr = slices.iter().collect();
        check_props(string.as_str(), smol)?;
    }
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Serialize, Deserialize)]
    struct SmolStrStruct {
        pub(crate) s: SmolStr,
        pub(crate) vec: Vec<SmolStr>,
        pub(crate) map: HashMap<SmolStr, SmolStr>,
    }

    #[test]
    fn test_serde() {
        let s = SmolStr::new("Hello, World");
        let s = serde_json::to_string(&s).unwrap();
        assert_eq!(s, "\"Hello, World\"");
        let s: SmolStr = serde_json::from_str(&s).unwrap();
        assert_eq!(s, "Hello, World");
    }

    #[test]
    fn test_serde_reader() {
        let s = SmolStr::new("Hello, World");
        let s = serde_json::to_string(&s).unwrap();
        assert_eq!(s, "\"Hello, World\"");
        let s: SmolStr = serde_json::from_reader(std::io::Cursor::new(s)).unwrap();
        assert_eq!(s, "Hello, World");
    }

    #[test]
    fn test_serde_struct() {
        let mut map = HashMap::new();
        map.insert(SmolStr::new("a"), SmolStr::new("ohno"));
        let struct_ = SmolStrStruct {
            s: SmolStr::new("Hello, World"),
            vec: vec![SmolStr::new("Hello, World"), SmolStr::new("Hello, World")],
            map,
        };
        let s = serde_json::to_string(&struct_).unwrap();
        let _new_struct: SmolStrStruct = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn test_serde_struct_reader() {
        let mut map = HashMap::new();
        map.insert(SmolStr::new("a"), SmolStr::new("ohno"));
        let struct_ = SmolStrStruct {
            s: SmolStr::new("Hello, World"),
            vec: vec![SmolStr::new("Hello, World"), SmolStr::new("Hello, World")],
            map,
        };
        let s = serde_json::to_string(&struct_).unwrap();
        let _new_struct: SmolStrStruct = serde_json::from_reader(std::io::Cursor::new(s)).unwrap();
    }

    #[test]
    fn test_serde_hashmap() {
        let mut map = HashMap::new();
        map.insert(SmolStr::new("a"), SmolStr::new("ohno"));
        let s = serde_json::to_string(&map).unwrap();
        let _s: HashMap<SmolStr, SmolStr> = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn test_serde_hashmap_reader() {
        let mut map = HashMap::new();
        map.insert(SmolStr::new("a"), SmolStr::new("ohno"));
        let s = serde_json::to_string(&map).unwrap();
        let _s: HashMap<SmolStr, SmolStr> =
            serde_json::from_reader(std::io::Cursor::new(s)).unwrap();
    }

    #[test]
    fn test_serde_vec() {
        let vec = vec![SmolStr::new(""), SmolStr::new("b")];
        let s = serde_json::to_string(&vec).unwrap();
        let _s: Vec<SmolStr> = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn test_serde_vec_reader() {
        let vec = vec![SmolStr::new(""), SmolStr::new("b")];
        let s = serde_json::to_string(&vec).unwrap();
        let _s: Vec<SmolStr> = serde_json::from_reader(std::io::Cursor::new(s)).unwrap();
    }
}

#[test]
fn test_search_in_hashmap() {
    let mut m = ::std::collections::HashMap::<SmolStr, i32>::new();
    m.insert("aaa".into(), 17);
    assert_eq!(17, *m.get("aaa").unwrap());
}

#[test]
fn test_from_char_iterator() {
    let examples = [
        // Simple keyword-like strings
        ("if", false),
        ("for", false),
        ("impl", false),
        // Strings containing two-byte characters
        ("パーティーへ行かないか", true),
        ("パーティーへ行か", true),
        ("パーティーへ行_", false),
        ("和製漢語", false),
        ("部落格", false),
        ("사회과학원 어학연구소", true),
        // String containing diverse characters
        ("表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀", true),
    ];
    for (raw, is_heap) in &examples {
        let s: SmolStr = raw.chars().collect();
        assert_eq!(s.as_str(), *raw);
        assert_eq!(s.is_heap_allocated(), *is_heap);
    }
    // String which has too many characters to even consider inlining: Chars::size_hint uses
    // (`len` + 3) / 4. With `len` = 89, this results in 23, so `from_iter` will immediately
    // heap allocate
    let raw = "a".repeat(23 * 4 + 1);
    let s: SmolStr = raw.chars().collect();
    assert_eq!(s.as_str(), raw);
    assert!(s.is_heap_allocated());
}

#[test]
fn test_bad_size_hint_char_iter() {
    struct BadSizeHint<I>(I);

    impl<T, I: Iterator<Item = T>> Iterator for BadSizeHint<I> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (1024, None)
        }
    }

    let data = "testing";
    let collected: SmolStr = BadSizeHint(data.chars()).collect();
    let new = SmolStr::new(data);

    assert!(!collected.is_heap_allocated());
    assert!(!new.is_heap_allocated());
    assert_eq!(new, collected);
}

#[test]
fn test_to_smolstr() {
    use smol_str::ToSmolStr;

    for i in 0..26 {
        let a = &"abcdefghijklmnopqrstuvwxyz"[i..];

        assert_eq!(a, a.to_smolstr());
        assert_eq!(a, smol_str::format_smolstr!("{}", a));
    }
}

#[test]
fn test_builder_push_str() {
    //empty
    let builder = SmolStrBuilder::new();
    assert_eq!("", builder.finish());

    // inline push
    let mut builder = SmolStrBuilder::new();
    builder.push_str("a");
    builder.push_str("b");
    let s = builder.finish();
    assert!(!s.is_heap_allocated());
    assert_eq!("ab", s);

    // inline max push
    let mut builder = SmolStrBuilder::new();
    builder.push_str(&"a".repeat(23));
    let s = builder.finish();
    assert!(!s.is_heap_allocated());
    assert_eq!("a".repeat(23), s);

    // heap push immediate
    let mut builder = SmolStrBuilder::new();
    builder.push_str(&"a".repeat(24));
    let s = builder.finish();
    assert!(s.is_heap_allocated());
    assert_eq!("a".repeat(24), s);

    // heap push succession
    let mut builder = SmolStrBuilder::new();
    builder.push_str(&"a".repeat(23));
    builder.push_str(&"a".repeat(23));
    let s = builder.finish();
    assert!(s.is_heap_allocated());
    assert_eq!("a".repeat(46), s);

    // heap push on multibyte char
    let mut builder = SmolStrBuilder::new();
    builder.push_str("ohnonononononononono!");
    builder.push('🤯');
    let s = builder.finish();
    assert!(s.is_heap_allocated());
    assert_eq!("ohnonononononononono!🤯", s);
}

#[test]
fn test_builder_push() {
    //empty
    let builder = SmolStrBuilder::new();
    assert_eq!("", builder.finish());

    // inline push
    let mut builder = SmolStrBuilder::new();
    builder.push('a');
    builder.push('b');
    let s = builder.finish();
    assert!(!s.is_heap_allocated());
    assert_eq!("ab", s);

    // inline max push
    let mut builder = SmolStrBuilder::new();
    for _ in 0..23 {
        builder.push('a');
    }
    let s = builder.finish();
    assert!(!s.is_heap_allocated());
    assert_eq!("a".repeat(23), s);

    // heap push
    let mut builder = SmolStrBuilder::new();
    for _ in 0..24 {
        builder.push('a');
    }
    let s = builder.finish();
    assert!(s.is_heap_allocated());
    assert_eq!("a".repeat(24), s);
}

#[cfg(test)]
mod test_str_ext {
    use smol_str::StrExt;

    #[test]
    fn large() {
        let lowercase = "aaaaaaAAAAAaaaaaaaaaaaaaaaaaaaaaAAAAaaaaaaaaaaaaaa".to_lowercase_smolstr();
        assert_eq!(lowercase, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(lowercase.is_heap_allocated());
    }

    #[test]
    fn to_lowercase() {
        let lowercase = "aßΔC".to_lowercase_smolstr();
        assert_eq!(lowercase, "aßδc");
        assert!(!lowercase.is_heap_allocated());
    }

    #[test]
    fn to_uppercase() {
        let uppercase = "aßΔC".to_uppercase_smolstr();
        assert_eq!(uppercase, "ASSΔC");
        assert!(!uppercase.is_heap_allocated());
    }

    #[test]
    fn to_ascii_lowercase() {
        let uppercase = "aßΔC".to_ascii_lowercase_smolstr();
        assert_eq!(uppercase, "aßΔc");
        assert!(!uppercase.is_heap_allocated());
    }

    #[test]
    fn to_ascii_uppercase() {
        let uppercase = "aßΔC".to_ascii_uppercase_smolstr();
        assert_eq!(uppercase, "AßΔC");
        assert!(!uppercase.is_heap_allocated());
    }

    #[test]
    fn replace() {
        let result = "foo_bar_baz".replace_smolstr("ba", "do");
        assert_eq!(result, "foo_dor_doz");
        assert!(!result.is_heap_allocated());
    }

    #[test]
    fn replacen() {
        let result = "foo_bar_baz".replacen_smolstr("ba", "do", 1);
        assert_eq!(result, "foo_dor_baz");
        assert!(!result.is_heap_allocated());
    }

    #[test]
    fn replacen_1_ascii() {
        let result = "foo_bar_baz".replacen_smolstr("o", "u", 1);
        assert_eq!(result, "fuo_bar_baz");
        assert!(!result.is_heap_allocated());
    }
}

#[cfg(feature = "borsh")]
mod borsh_tests {
    use borsh::BorshDeserialize;
    use smol_str::{SmolStr, ToSmolStr};
    use std::io::Cursor;

    #[test]
    fn borsh_serialize_stack() {
        let smolstr_on_stack = "aßΔCaßδc".to_smolstr();
        let mut buffer = Vec::new();
        borsh::BorshSerialize::serialize(&smolstr_on_stack, &mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        let decoded: SmolStr = borsh::BorshDeserialize::deserialize_reader(&mut cursor).unwrap();
        assert_eq!(smolstr_on_stack, decoded);
    }
    #[test]
    fn borsh_serialize_heap() {
        let smolstr_on_heap = "aßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδcaßΔCaßδc".to_smolstr();
        let mut buffer = Vec::new();
        borsh::BorshSerialize::serialize(&smolstr_on_heap, &mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        let decoded: SmolStr = borsh::BorshDeserialize::deserialize_reader(&mut cursor).unwrap();
        assert_eq!(smolstr_on_heap, decoded);
    }
    #[test]
    fn borsh_non_utf8_stack() {
        let invalid_utf8: Vec<u8> = vec![0xF0, 0x9F, 0x8F]; // Incomplete UTF-8 sequence

        let wrong_utf8 = SmolStr::from(unsafe { String::from_utf8_unchecked(invalid_utf8) });
        let mut buffer = Vec::new();
        borsh::BorshSerialize::serialize(&wrong_utf8, &mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        let result = SmolStr::deserialize_reader(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn borsh_non_utf8_heap() {
        let invalid_utf8: Vec<u8> = vec![
            0xC1, 0x8A, 0x5F, 0xE2, 0x3A, 0x9E, 0x3B, 0xAA, 0x01, 0x08, 0x6F, 0x2F, 0xC0, 0x32,
            0xAB, 0xE1, 0x9A, 0x2F, 0x4A, 0x3F, 0x25, 0x0D, 0x8A, 0x2A, 0x19, 0x11, 0xF0, 0x7F,
            0x0E, 0x80,
        ];
        let wrong_utf8 = SmolStr::from(unsafe { String::from_utf8_unchecked(invalid_utf8) });
        let mut buffer = Vec::new();
        borsh::BorshSerialize::serialize(&wrong_utf8, &mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        let result = SmolStr::deserialize_reader(&mut cursor);
        assert!(result.is_err());
    }
}
