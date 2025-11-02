#![expect(clippy::print_literal)]
#![warn(clippy::print_stdout)]

fn main() {
    println!("Hello");
    //~^ print_stdout

    print!("Hello");
    //~^ print_stdout

    print!("Hello {}", "World");
    //~^ print_stdout

    print!("Hello {:?}", "World");
    //~^ print_stdout

    print!("Hello {:#?}", "#orld");
    //~^ print_stdout

    assert_eq!(42, 1337);

    vec![1, 2];
}
