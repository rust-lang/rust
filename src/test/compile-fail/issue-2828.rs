struct NoCopy {
    n: int
}
fn NoCopy() -> NoCopy {
    NoCopy { n: 0 }
}

impl NoCopy: Drop {
    fn finalize(&self) {
        log(error, "running destructor");
    }
}

fn main() {
    let x = NoCopy();

    let f = fn~() { assert x.n == 0; }; //~ ERROR copying a noncopyable value
    let g = copy f;

    f(); g();
}