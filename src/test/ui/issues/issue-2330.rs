enum chan { }

trait channel<T> {
    fn send(&self, v: T);
}

// `chan` is not a trait, it's an enum
impl chan for isize { //~ ERROR expected trait, found enum `chan`
    fn send(&self, v: isize) { panic!() }
}

fn main() {
}
