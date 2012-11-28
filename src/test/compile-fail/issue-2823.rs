struct C {
    x: int,
}

impl C : Drop {
    fn finalize(&self) {
        error!("dropping: %?", self.x);
    }
}

fn main() {
    let c = C{ x: 2};
    let d = copy c; //~ ERROR copying a noncopyable value
    error!("%?", d.x);
}
