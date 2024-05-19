#[warn(clippy::exit)]

fn also_not_main() {
    std::process::exit(3);
    //~^ ERROR: usage of `process::exit`
    //~| NOTE: `-D clippy::exit` implied by `-D warnings`
}

fn main() {
    if true {
        std::process::exit(2);
    };
    also_not_main();
    std::process::exit(1);
}
