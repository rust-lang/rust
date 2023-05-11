#[inline(unknown)] //~ ERROR E0535
pub fn something() {}

fn main() {
    something();
}
