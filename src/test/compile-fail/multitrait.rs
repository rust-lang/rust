struct S {
 y: int
}

impl S: Cmp, ToStr { //~ ERROR: expected `{` but found `,`
  fn eq(&&other: S) { false }
  fn to_str() -> ~str { ~"hi" }
}