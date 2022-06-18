#[warn(clippy::blacklisted_name)]

fn main() {
    // `foo` is part of the default configuration
    let foo = "bar";
    // `ducks` was unrightfully blacklisted
    let ducks = ["quack", "quack"];
    // `fox` is okay
    let fox = ["what", "does", "the", "fox", "say", "?"];
}
