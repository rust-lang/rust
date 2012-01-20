

fn foo<T>(o: myoption<T>) -> int {
    let x: int = 5;
    alt o { none::<T> { } some::<T>(t) { x += 1; } }
    ret x;
}

enum myoption<T> { none; some(T); }

fn main() { log(debug, 5); }
