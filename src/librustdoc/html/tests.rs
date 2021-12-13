use crate::html::format::href_relative_parts;

fn assert_relative_path(expected: &str, relative_to_fqp: &[&str], fqp: &[&str]) {
    let relative_to_fqp: Vec<String> = relative_to_fqp.iter().copied().map(String::from).collect();
    let fqp: Vec<String> = fqp.iter().copied().map(String::from).collect();
    assert_eq!(expected, href_relative_parts(&fqp, &relative_to_fqp).finish());
}

#[test]
fn href_relative_parts_basic() {
    let relative_to_fqp = &["std", "vec"];
    let fqp = &["std", "iter"];
    assert_relative_path("../iter", relative_to_fqp, fqp);
}
#[test]
fn href_relative_parts_parent_module() {
    let relative_to_fqp = &["std", "vec"];
    let fqp = &["std"];
    assert_relative_path("..", relative_to_fqp, fqp);
}
#[test]
fn href_relative_parts_different_crate() {
    let relative_to_fqp = &["std", "vec"];
    let fqp = &["core", "iter"];
    assert_relative_path("../../core/iter", relative_to_fqp, fqp);
}
#[test]
fn href_relative_parts_same_module() {
    let relative_to_fqp = &["std", "vec"];
    let fqp = &["std", "vec"];
    assert_relative_path("", relative_to_fqp, fqp);
}
#[test]
fn href_relative_parts_child_module() {
    let relative_to_fqp = &["std"];
    let fqp = &["std", "vec"];
    assert_relative_path("vec", relative_to_fqp, fqp);
}
#[test]
fn href_relative_parts_root() {
    let relative_to_fqp = &[];
    let fqp = &["std"];
    assert_relative_path("std", relative_to_fqp, fqp);
}
