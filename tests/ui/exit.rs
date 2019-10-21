#[warn(clippy::exit)]

fn not_main() {
    if true {
        std::process::exit(4);
    }
}

fn also_not_main() {
    std::process::exit(3);
}

fn main() {
    if true {
        std::process::exit(2);
    };
    also_not_main();
    not_main();
    std::process::exit(1);
}
