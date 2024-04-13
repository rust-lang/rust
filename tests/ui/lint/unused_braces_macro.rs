//@ build-pass
pub fn foo<const BAR: bool> () {}

fn main() {
    foo::<{cfg!(FALSE)}>();
}
