#[inline()] //~ ERROR E0534
pub fn something() {}

fn main() {
    something();
}
