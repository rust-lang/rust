#[inline(unknown)] //~ ERROR malformed `inline` attribute
pub fn something() {}

fn main() {
    something();
}
