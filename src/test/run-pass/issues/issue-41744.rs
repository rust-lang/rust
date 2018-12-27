// run-pass
trait Tc {}
impl Tc for bool {}

fn main() {
    let _: &[&Tc] = &[&true];
}
