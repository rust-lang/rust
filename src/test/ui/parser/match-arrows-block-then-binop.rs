// compile-flags: -Z parse-only

fn main() {

    match 0 {
      0 => {
      } + 5 //~ ERROR expected pattern, found `+`
    }
}
