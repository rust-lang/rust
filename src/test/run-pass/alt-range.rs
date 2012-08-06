fn main() {
    match 5u {
      1u to 5u => {}
      _ => fail ~"should match range",
    }
    match 5u {
      6u to 7u => fail ~"shouldn't match range",
      _ => {}
    }
    match check 5u {
      1u => fail ~"should match non-first range",
      2u to 6u => {}
    }
    match 'c' {
      'a' to 'z' => {}
      _ => fail ~"should suppport char ranges"
    }
    match -3 {
      -7 to 5 => {}
      _ => fail ~"should match signed range"
    }
    match 3.0 {
      1.0 to 5.0 => {}
      _ => fail ~"should match float range"
    }
    match -1.5 {
      -3.6 to 3.6 => {}
      _ => fail ~"should match negative float range"
    }
}
