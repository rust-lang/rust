// run-pass

fn main() {
    assert_eq!(false, match -50_i8 { -128i8..=-101i8 => true, _ => false, });

    assert_eq!(false, if let -128i8..=-101i8 = -50_i8 { true } else { false });
}
