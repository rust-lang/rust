fn test_allow_on_function() {
    #![allow(clippy::disallowed_macros)]

    panic!();
    panic!("message");
    panic!("{}", 123);
    panic!("{} {}", 123, 456);
    println!("test");
    println!("{}", 123);
    vec![1, 2, 3];
}

fn test_expect_on_function() {
    #![expect(clippy::disallowed_macros)]

    panic!();
    panic!("message");
    panic!("{}", 123);
    panic!("{} {}", 123, 456);
    println!("test");
    println!("{}", 123);
    vec![1, 2, 3];
}

fn test_no_attributes() {
    panic!();
    //~^ disallowed_macros
    panic!("message");
    //~^ disallowed_macros
    panic!("{}", 123);
    //~^ disallowed_macros
    panic!("{} {}", 123, 456);
    //~^ disallowed_macros
    println!("test");
    //~^ disallowed_macros
    println!("{}", 123);
    //~^ disallowed_macros
    vec![1, 2, 3];
    //~^ disallowed_macros
}

#[allow(clippy::disallowed_macros)]
mod allowed_module {
    pub fn test() {
        panic!();
        panic!("message");
        panic!("{}", 123);
        println!("test");
        vec![1, 2, 3];
    }
}

#[expect(clippy::disallowed_macros)]
mod expected_module {
    pub fn test() {
        panic!();
        panic!("message");
        panic!("{}", 123);
        println!("test");
        vec![1, 2, 3];
    }
}

fn main() {
    test_allow_on_function();
    test_expect_on_function();
    test_no_attributes();
    allowed_module::test();
    expected_module::test();
}
