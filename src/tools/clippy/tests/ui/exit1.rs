#![warn(clippy::exit)]

fn not_main() {
    if true {
        std::process::exit(4);
        //~^ ERROR: usage of `process::exit`
        //~| NOTE: `-D clippy::exit` implied by `-D warnings`
    }
}

fn main() {
    if true {
        std::process::exit(2);
    };
    not_main();
    std::process::exit(1);
}
