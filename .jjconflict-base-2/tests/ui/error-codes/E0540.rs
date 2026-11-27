#[inline()] //~ ERROR malformed `inline` attribute input
pub fn something() {}

fn main() {
    something();
}
