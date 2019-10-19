#[warn(clippy::exit)]
fn not_main() {
    std::process::exit(3);
}

fn main() {
    if true {
        std::process::exit(2);
    };
    not_main();
    std::process::exit(1);
}
