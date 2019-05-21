// revisions:rpass1 rpass2

struct A;

#[cfg(rpass2)]
impl From<A> for () {
    fn from(_: A) {}
}

fn main() {}
