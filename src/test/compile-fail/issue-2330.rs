enum chan { }

iface channel<T> {
    fn send(v: T);
}

// `chan` is not an iface, it's an enum
impl of chan for int { //~ ERROR can only implement interface types
    fn send(v: int) { fail }
}

fn main() {
}
