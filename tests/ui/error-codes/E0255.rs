use bar::foo;

fn foo() {} //~ ERROR E0255

mod bar {
     pub fn foo() {}
}

fn main() {}
