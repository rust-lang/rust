#![warn(clippy::exit)]

fn not_main() {
    if true {
        std::process::exit(4);
        //~^ exit
    }
}

fn main() {
    if true {
        std::process::exit(2);
    };
    not_main();
    std::process::exit(1);
}
