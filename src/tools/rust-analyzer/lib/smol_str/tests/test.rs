extern crate smol_str;
extern crate serde_json;
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

fn check_props(s: &str) -> Result<(), proptest::test_runner::TestCaseError> {
    let smol = SmolStr::new(s);
    prop_assert_eq!(smol.as_str(), s);
    prop_assert_eq!(smol.len(), s.len());
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

#[test]
fn test_serde() {
    let s = SmolStr::new("Hello, World");
    let s = serde_json::to_string(&s).unwrap();
    assert_eq!(s, "\"Hello, World\"");
    let s: SmolStr = serde_json::from_str(&s).unwrap();
    assert_eq!(s, "Hello, World");
}
