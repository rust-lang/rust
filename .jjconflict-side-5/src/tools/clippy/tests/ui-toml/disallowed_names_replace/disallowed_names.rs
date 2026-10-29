#![warn(clippy::disallowed_names)]

fn main() {
    // `foo` is part of the default configuration
    let foo = "bar";
    // `ducks` was unrightfully disallowed
    let ducks = ["quack", "quack"];
    //~^ disallowed_names
    // `fox` is okay
    let fox = ["what", "does", "the", "fox", "say", "?"];
}
