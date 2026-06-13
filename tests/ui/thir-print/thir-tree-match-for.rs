//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree
//@ normalize-stdout: "\[[a-z0-9]{4}\]" -> ""
//@ normalize-stdout: "DefId\(\d+:\d+" -> "DefId(N:M"
//@ normalize-stdout: "ReprOptions\s*\{[^}]+\}" -> "ReprOptions {}"

fn match_non_loop(x:u32, y: u32) {
  match x {
    0 | 1 => 0,
    2..3  => 1,
       y  => 2,
  };
}

fn match_from_for(x: u32, y: u32) {
  for i in x..y {
    i;
  }
}

// same resulting structure
fn match_loop_nonfor(x: u32, y: u32) {
  match IntoIterator::into_iter(x..y) {
    mut iter => loop {
      match Iterator::next(&mut iter) {
          Option::None => break,
          Option::Some(i) => { i; },
      };
    }
  }
}

fn main() {}
