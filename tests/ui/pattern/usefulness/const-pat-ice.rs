// check-pass

const FOO: &&&u32 = &&&42;

fn main() {
    match unimplemented!() {
        &&&42 => {},
        FOO => {},
        _ => {},
    }
}
