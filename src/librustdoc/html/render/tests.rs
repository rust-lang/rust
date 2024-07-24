use std::cmp::Ordering;

use super::print_item::compare_names;
use super::{AllTypes, Buffer};

#[test]
fn test_compare_names() {
    for &(a, b) in &[
        ("hello", "world"),
        ("", "world"),
        ("123", "hello"),
        ("123", ""),
        ("123test", "123"),
        ("hello", ""),
        ("hello", "hello"),
        ("hello123", "hello123"),
        ("hello123", "hello12"),
        ("hello12", "hello123"),
        ("hello01abc", "hello01xyz"),
        ("hello0abc", "hello0"),
        ("hello0", "hello0abc"),
        ("01", "1"),
    ] {
        assert_eq!(compare_names(a, b), a.cmp(b), "{:?} - {:?}", a, b);
    }
    assert_eq!(compare_names("u8", "u16"), Ordering::Less);
    assert_eq!(compare_names("u32", "u16"), Ordering::Greater);
    assert_eq!(compare_names("u8_to_f64", "u16_to_f64"), Ordering::Less);
    assert_eq!(compare_names("u32_to_f64", "u16_to_f64"), Ordering::Greater);
    assert_eq!(compare_names("u16_to_f64", "u16_to_f64"), Ordering::Equal);
    assert_eq!(compare_names("u16_to_f32", "u16_to_f64"), Ordering::Less);
}

#[test]
fn test_name_sorting() {
    let names = [
        "Apple", "Banana", "Fruit", "Fruit0", "Fruit00", "Fruit01", "Fruit02", "Fruit1", "Fruit2",
        "Fruit20", "Fruit30x", "Fruit100", "Pear",
    ];
    let mut sorted = names.to_owned();
    sorted.sort_by(|&l, r| compare_names(l, r));
    assert_eq!(names, sorted);
}

#[test]
fn test_all_types_prints_header_once() {
    // Regression test for #82477
    let all_types = AllTypes::new();

    let mut buffer = Buffer::new();
    all_types.print(&mut buffer);

    assert_eq!(1, buffer.into_inner().matches("List of all items").count());
}
