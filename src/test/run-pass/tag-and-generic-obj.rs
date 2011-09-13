

tag colour { red; green; }

obj foo<T>() {
    fn meth(x: T) { }
}

fn main() { foo::<colour>().meth(red); }
