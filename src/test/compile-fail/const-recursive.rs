// error-pattern: recursive constant
const a: int = b;
const b: int = a;

fn main() {
}