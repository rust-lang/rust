//@ check-pass

#![warn(redundant_imports)]

mod foo {
    use std::fmt;

    pub struct String;

    impl String {
        pub fn new() -> String {
            String{}
        }
    }

    impl fmt::Display for String {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "String")
        }
    }
}

fn main() {

    {
        use std::string::String; //~ WARNING the item `String` is imported redundantly
        // 'String' from 'std::string::String'.
        let s = String::new();
        println!("{}", s);
    }

    {
        // 'String' from 'std::string::String'.
        let s = String::new();
        println!("{}", s);
    }

    {
        use foo::*;
        // 'String' from 'foo::String'.
        let s = String::new();
        println!("{}", s);
    }

}
