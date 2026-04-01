//@ run-pass
#![allow(non_camel_case_types)]
#![allow(dead_code)]

#[derive(Debug)]
enum a_tag {
    a_tag_var(u64)
}

#[derive(Debug)]
struct t_rec {
    c8: u8,
    t: a_tag
}

pub fn main() {
    let x = t_rec {c8: 22, t: a_tag::a_tag_var(44)};
    let y = format!("{:?}", x);
    println!("y = {:?}", y);
    assert_eq!(y, "t_rec { c8: 22, t: a_tag_var(44) }".to_string());
}
