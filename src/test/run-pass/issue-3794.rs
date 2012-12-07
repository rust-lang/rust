// xfail-test
trait T {
    fn print(&self);
}

struct S {
    s: int,
}

impl S: T {
    fn print(&self) {
        io::println(fmt!("%?", self));
    }
}

fn print_t(t: &T) {
    t.print();
}

fn print_s(s: &S) {
    s.print();
}

fn main() {
    let s: @S = @S { s: 5 };
    print_s(s);
    let t: @T = s as @T;
    print_t(t);

}
