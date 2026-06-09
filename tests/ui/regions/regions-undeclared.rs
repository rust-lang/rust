static c_x: &'blk isize = &22; //~ ERROR use of undeclared lifetime name `'blk`

enum EnumDecl {
    Foo(&'a isize), //~ ERROR use of undeclared lifetime name `'a`
    Bar(&'a isize), //~ ERROR use of undeclared lifetime name `'a`
}

fn fnDecl(x: &'a isize, //~ ERROR use of undeclared lifetime name `'a`
          y: &'a isize) //~ ERROR use of undeclared lifetime name `'a`
{}

fn main() {
}
