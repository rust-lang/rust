// Issue #53

fn main() {
    match check ~"test" { ~"not-test" => fail, ~"test" => (), _ => fail }

    enum t { tag1(~str), tag2, }


    match tag1(~"test") {
      tag2 => fail,
      tag1(~"not-test") => fail,
      tag1(~"test") => (),
      _ => fail
    }

    let x = match check ~"a" { ~"a" => 1, ~"b" => 2 };
    assert (x == 1);

    match check ~"a" { ~"a" => { } ~"b" => { } }

}
