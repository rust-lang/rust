pub mod foo {
    pub mod bar {
        pub struct A;
    }
}

pub struct Foo {
  a: Vec<foo::bar:A>,
  //~^ ERROR expected
  //~| HELP path separator
}

fn main() {}
