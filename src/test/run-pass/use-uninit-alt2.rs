

fn foo<T>(o: myoption<T>) -> int {
    let x: int;
    alt o { none::<T> { fail; } some::<T>(t) { x = 5; } }
    ret x;
}

enum myoption<T> { none; some(T); }

fn main() { log(debug, 5); }
