struct noncopyable {
    i: (),
}

impl noncopyable : Drop {
    fn finalize(&self) {
        error!("dropped");
    }
}

fn noncopyable() -> noncopyable {
    noncopyable {
        i: ()
    }
}

enum wrapper = noncopyable;

fn main() {
    let x1 = wrapper(noncopyable());
    let _x2 = move *x1; //~ ERROR moving out of enum content
}
