struct S {
 y: isize
}

impl Cmp, ToString for S {
//~^ ERROR: expected one of `!`, `(`, `+`, `::`, `<`, `for`, `where`, or `{`, found `,`
  fn eq(&&other: S) { false }
  fn to_string(&self) -> String { "hi".to_string() }
}
