use super::*;

#[test]
fn test_name_key() {
    assert_eq!(name_key("0"), ("", 0, 1));
    assert_eq!(name_key("123"), ("", 123, 0));
    assert_eq!(name_key("Fruit"), ("Fruit", 0, 0));
    assert_eq!(name_key("Fruit0"), ("Fruit", 0, 1));
    assert_eq!(name_key("Fruit0000"), ("Fruit", 0, 4));
    assert_eq!(name_key("Fruit01"), ("Fruit", 1, 1));
    assert_eq!(name_key("Fruit10"), ("Fruit", 10, 0));
    assert_eq!(name_key("Fruit123"), ("Fruit", 123, 0));
}

#[test]
fn test_name_sorting() {
    let names = ["Apple",
                 "Banana",
                 "Fruit", "Fruit0", "Fruit00",
                 "Fruit1", "Fruit01",
                 "Fruit2", "Fruit02",
                 "Fruit20",
                 "Fruit30x",
                 "Fruit100",
                 "Pear"];
    let mut sorted = names.to_owned();
    sorted.sort_by_key(|&s| name_key(s));
    assert_eq!(names, sorted);
}
