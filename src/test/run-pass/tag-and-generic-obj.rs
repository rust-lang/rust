

tag colour { red; green; }

obj foo[T]() {
    fn meth(&T x) { }
}

fn main() { foo[colour]().meth(red); }