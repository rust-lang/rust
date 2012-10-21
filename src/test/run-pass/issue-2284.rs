// xfail-test
trait Send {
    fn f();
}

fn f<T: Send>(t: T) {
    t.f();
}

fn main() {
}