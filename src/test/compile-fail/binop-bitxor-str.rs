// xfail-stage0
// error-pattern:^ cannot be applied to type `str`

fn main() {
  auto x = "a" ^ "b";
}