#![feature(plugin)]
#![plugin(clippy)]

#![deny(string_add_assign)]

fn main() {
    let x = "".to_owned();

    for i in (1..3) {
        x = x + "."; //~ERROR
    }
}
