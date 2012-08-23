// Issue #53

fn main() {
    match ~"test" { ~"not-test" => fail, ~"test" => (), _ => fail }

    enum t { tag1(~str), tag2, }


    match tag1(~"test") {
      tag2 => fail,
      tag1(~"not-test") => fail,
      tag1(~"test") => (),
      _ => fail
    }

    let x = match ~"a" { ~"a" => 1, ~"b" => 2, _ => fail };
    assert (x == 1);

    match ~"a" { ~"a" => { } ~"b" => { }, _ => fail }

}
