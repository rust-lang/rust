static s: int = 1;
static e: int = 42;

fn main() {
    match 7 {
        s..e => (),
        _ => (),
    }
}
