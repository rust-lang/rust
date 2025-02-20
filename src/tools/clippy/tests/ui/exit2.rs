#![warn(clippy::exit)]

fn also_not_main() {
    std::process::exit(3);
    //~^ exit
}

fn main() {
    if true {
        std::process::exit(2);
    };
    also_not_main();
    std::process::exit(1);
}
