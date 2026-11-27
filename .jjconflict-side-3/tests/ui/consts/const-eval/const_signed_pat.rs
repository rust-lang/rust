//@ check-pass

fn main() {
    const MIN: i8 = -5;
    match 5i8 {
        MIN..=-1 => {},
        _ => {},
    }
}
