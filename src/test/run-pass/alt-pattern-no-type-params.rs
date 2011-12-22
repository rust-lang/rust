tag maybe<T> { nothing; just(T); }

fn foo(x: maybe<int>) {
    alt x { nothing. { #error("A"); } just(a) { #error("B"); } }
}

fn main() { }
