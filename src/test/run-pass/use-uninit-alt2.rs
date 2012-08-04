

fn foo<T>(o: myoption<T>) -> int {
    let mut x: int;
    alt o { none::<T> => { fail; } some::<T>(t) => { x = 5; } }
    return x;
}

enum myoption<T> { none, some(T), }

fn main() { log(debug, 5); }
