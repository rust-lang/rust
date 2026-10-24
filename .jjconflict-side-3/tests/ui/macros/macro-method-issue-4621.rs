//@ run-pass

struct A;

macro_rules! make_thirteen_method {() => (fn thirteen(&self)->isize {13})}
impl A { make_thirteen_method!(); }

fn main() {
    assert_eq!(A.thirteen(),13);
}
