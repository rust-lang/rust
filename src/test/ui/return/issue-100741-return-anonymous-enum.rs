struct Foo {}

fn foo<'a>() -> i32 | Vec<i32> | &str | &'a String | Foo {
    //~^ ERROR: anonymous enums are not supported
}

fn main() {}
