enum chan { }

trait channel<T> {
    fn send(v: T);
}

// `chan` is not a trait, it's an enum
impl of chan for int { //~ ERROR can only implement trait types
    fn send(v: int) { fail }
}

fn main() {
}
