fn f<T:Eq + Ord>(_: T) {
}

pub fn main() {
    f(3);
}
