pub mod foo {
    pub mod bar {
        pub struct A;
    }
}

pub struct Foo {
  a: Vec<foo::bar:A>,
  //~^ ERROR expected
  //~| HELP you might have meant to write a path
}

fn main() {}
