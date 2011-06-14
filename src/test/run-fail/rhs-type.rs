// Tests that trans treats the rhs of pth's decl
// as a _|_-typed thing, not a str-typed thing
// xfail-stage0
// error-pattern:bye
fn main() {
  auto pth = fail "bye";

  let rec(str t) res = rec(t=pth);

}
