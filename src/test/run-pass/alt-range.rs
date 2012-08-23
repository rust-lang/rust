fn main() {
    match 5u {
      1u..5u => {}
      _ => fail ~"should match range",
    }
    match 5u {
      6u..7u => fail ~"shouldn't match range",
      _ => {}
    }
    match 5u {
      1u => fail ~"should match non-first range",
      2u..6u => {}
      _ => fail ~"math is broken"
    }
    match 'c' {
      'a'..'z' => {}
      _ => fail ~"should suppport char ranges"
    }
    match -3 {
      -7..5 => {}
      _ => fail ~"should match signed range"
    }
    match 3.0 {
      1.0..5.0 => {}
      _ => fail ~"should match float range"
    }
    match -1.5 {
      -3.6..3.6 => {}
      _ => fail ~"should match negative float range"
    }
}
