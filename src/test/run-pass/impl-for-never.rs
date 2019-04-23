// Test that we can call static methods on ! both directly and when it appears in a generic

#![feature(never_type)]

trait StringifyType {
    fn stringify_type() -> &'static str;
}

impl StringifyType for ! {
    fn stringify_type() -> &'static str {
        "!"
    }
}

fn maybe_stringify<T: StringifyType>(opt: Option<T>) -> &'static str {
    match opt {
        Some(_) => T::stringify_type(),
        None => "none",
    }
}

fn main() {
    println!("! is {}", <!>::stringify_type());
    println!("None is {}", maybe_stringify(None::<!>));
}
