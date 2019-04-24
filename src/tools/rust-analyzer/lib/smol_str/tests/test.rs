extern crate serde_json;
extern crate smol_str;
#[macro_use]
extern crate proptest;

use smol_str::SmolStr;

#[test]
#[cfg(target_pointer_width = "64")]
fn smol_str_is_smol() {
    assert_eq!(
        ::std::mem::size_of::<SmolStr>(),
        ::std::mem::size_of::<String>(),
    );
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
    assert_eq!(s, "Hello, World!")
}

fn check_props(s: &str) -> Result<(), proptest::test_runner::TestCaseError> {
    let smol = SmolStr::new(s);
    prop_assert_eq!(smol.as_str(), s);
    prop_assert_eq!(smol.len(), s.len());
    prop_assert_eq!(smol.is_empty(), s.is_empty());
    Ok(())
}

proptest! {
    #[test]
    fn roundtrip(s: String) {
        check_props(s.as_str())?;
    }

    #[test]
    fn roundtrip_spaces(s in r"( )*") {
        check_props(s.as_str())?;
    }

    #[test]
    fn roundtrip_newlines(s in r"\n*") {
        check_props(s.as_str())?;
    }

    #[test]
    fn roundtrip_ws(s in r"( |\n)*") {
        check_props(s.as_str())?;
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    let s = SmolStr::new("Hello, World");
    let s = serde_json::to_string(&s).unwrap();
    assert_eq!(s, "\"Hello, World\"");
    let s: SmolStr = serde_json::from_str(&s).unwrap();
    assert_eq!(s, "Hello, World");
}

#[test]
fn test_search_in_hashmap() {
    let mut m = ::std::collections::HashMap::<SmolStr, i32>::new();
    m.insert("aaa".into(), 17);
    assert_eq!(17, *m.get("aaa").unwrap());
}

#[test]
fn test_from_iterator() {
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

        // String containin diverse characters
        ("表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀", true),
    ];
    for (raw, is_heap) in &examples {
        let s: SmolStr = raw.chars().collect();
        assert_eq!(s.as_str(), *raw);
        assert_eq!(s.is_heap_allocated(), *is_heap);
    }
}
