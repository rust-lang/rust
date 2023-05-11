// run-pass
#![allow(dead_code)]
// Issue #53
#![allow(non_camel_case_types)]


pub fn main() {
    match "test" { "not-test" => panic!(), "test" => (), _ => panic!() }

    enum t { tag1(String), tag2, }


    match t::tag1("test".to_string()) {
      t::tag2 => panic!(),
      t::tag1(ref s) if "test" != &**s => panic!(),
      t::tag1(ref s) if "test" == &**s => (),
      _ => panic!()
    }

    let x = match "a" { "a" => 1, "b" => 2, _ => panic!() };
    assert_eq!(x, 1);

    match "a" { "a" => { } "b" => { }, _ => panic!() }

}
