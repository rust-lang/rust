use super::*;

#[test]
fn test_try_from_invalid_version() {
    assert!("".parse::<Version>().is_err());
    assert!("hello".parse::<Version>().is_err());
    assert!("1.32.hi".parse::<Version>().is_err());
    assert!("1.32..1".parse::<Version>().is_err());
    assert!("1.32".parse::<Version>().is_err());
    assert!("1.32.0.1".parse::<Version>().is_err());
}

#[test]
fn test_try_from_single() {
    assert_eq!("1.32.0".parse(), Ok(Version::Explicit { parts: [1, 32, 0] }));
    assert_eq!("1.0.0".parse(), Ok(Version::Explicit { parts: [1, 0, 0] }));
}

#[test]
fn test_compare() {
    let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
    let v_1_32_0 = "1.32.0".parse::<Version>().unwrap();
    let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();
    assert!(v_1_0_0 < v_1_32_1);
    assert!(v_1_0_0 < v_1_32_0);
    assert!(v_1_32_0 < v_1_32_1);
}

#[test]
fn test_to_string() {
    let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
    let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();

    assert_eq!(v_1_0_0.to_string(), "1.0.0");
    assert_eq!(v_1_32_1.to_string(), "1.32.1");
    assert_eq!(format!("{v_1_32_1:<8}"), "1.32.1  ");
    assert_eq!(format!("{v_1_32_1:>8}"), "  1.32.1");
}
