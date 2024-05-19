#![warn(clippy::ineffective_open_options)]

use std::fs::OpenOptions;

fn main() {
    let file = OpenOptions::new()
        .create(true)
        .write(true) //~ ERROR: unnecessary use of `.write(true)`
        .append(true)
        .open("dump.json")
        .unwrap();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .write(true) //~ ERROR: unnecessary use of `.write(true)`
        .open("dump.json")
        .unwrap();

    // All the next calls are ok.
    let file = OpenOptions::new()
        .create(true)
        .write(false)
        .append(true)
        .open("dump.json")
        .unwrap();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .append(false)
        .open("dump.json")
        .unwrap();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(false)
        .append(false)
        .open("dump.json")
        .unwrap();
    let file = OpenOptions::new().create(true).append(true).open("dump.json").unwrap();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("dump.json")
        .unwrap();
}
