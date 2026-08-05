pub mod foo {
    pub mod bar {
        pub struct A;
    }
}

pub struct Foo {
  a: Vec<foo::bar:A>,
  //~^ ERROR path separator must be a double colon
  //~| HELP use a double colon instead
}

fn main() {}
