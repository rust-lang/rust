// xfail-fast - more windows randomness under check-fast

fn main() {
    alt 5u {
      1u to 5u {}
      _ { fail "should match range"; }
    }
    alt 5u {
      6u to 7u { fail "shouldn't match range"; }
      _ {}
    }
    alt 5u {
      1u { fail "should match non-first range"; }
      2u to 6u {}
    }
    alt 'c' {
      'a' to 'z' {}
      _ { fail "should suppport char ranges"; }
    }
    alt -3 {
      -7 to 5 {}
      _ { fail "should match signed range"; }
    }
    alt 3.0 {
      1.0 to 5.0 {}
      _ { fail "should match float range"; }
    }
    alt -1.5 {
      -3.6 to 3.6 {}
      _ { fail "should match negative float range"; }
    }
}
