// check-pass

pub trait Build {
    type Output<O>;
    fn build<O>(self, input: O) -> Self::Output<O>;
}

pub struct IdentityBuild;
impl Build for IdentityBuild {
    type Output<O> = O;
    fn build<O>(self, input: O) -> Self::Output<O> {
        input
    }
}

fn a() {
    let _x: u8 = IdentityBuild.build(10);
}

fn b() {
    let _x: Vec<u8> = IdentityBuild.build(Vec::new());
}

fn c() {
    let mut f = IdentityBuild.build(|| ());
    (f)();
}

pub fn main() {
    a();
    b();
    c();
}
