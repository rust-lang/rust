enum Chan { }

trait Channel<T> {
    fn send(&self, v: T);
}

// `Chan` is not a trait, it's an enum
impl Chan for isize { //~ ERROR expected trait, found enum `Chan`
    fn send(&self, v: isize) { panic!() }
}

fn main() {
}
