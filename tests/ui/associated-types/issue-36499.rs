//@ error-pattern: aborting due to 1 previous error

fn main() {
    2 + +2; //~ ERROR leading `+` is not supported
}
