pub fn foo<T>(s: S<T>, t: S<T>) {
    let _ = s == t; //~ ERROR binary operation `==` cannot be applied to type `S<T>`
}

struct S<T>(T);

fn main() {}
