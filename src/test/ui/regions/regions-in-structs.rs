struct Yes1<'a> {
  x: &'a usize,
}

struct Yes2<'a> {
  x: &'a usize,
}

struct StructDecl {
    a: &'a isize, //~ ERROR use of undeclared lifetime name `'a`
    b: &'a isize, //~ ERROR use of undeclared lifetime name `'a`
}


fn main() {}
