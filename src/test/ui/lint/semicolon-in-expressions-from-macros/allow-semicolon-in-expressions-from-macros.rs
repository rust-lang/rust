// check-pass
// Ensure that trailing semicolons are allowed by default

macro_rules! foo {
    () => {
        true;
    }
}

fn main() {
    let val = match true {
        true => false,
        _ => foo!()
    };
}
