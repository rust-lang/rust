pub fn foo<T>(s: S<T>, t: S<T>) {
    let _ = s == t; //~ ERROR: can't compare `S<T>`
}

struct S<T>(T);

fn main() {}
