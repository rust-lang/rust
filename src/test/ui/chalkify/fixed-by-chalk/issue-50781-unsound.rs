// compile-flags: -Z chalk-migration

pub struct Value {
    a: String,
}

pub struct Borrowed<'a> {
    b: &'a str,
}

pub fn parse(a: &Value) -> Borrowed<'_> {
    Borrowed { b: &a.a }
}

pub fn not<T>(predicate: impl Fn(&T) -> bool) -> impl Fn(&T) -> bool {
    move |t: &T| !predicate(t)
}

/// Transform a predicate on `Borrowed`s into a predicate for `Value`s
pub fn borrowed(predicate: impl for<'a> Fn(&Borrowed<'_>) -> bool) -> impl Fn(&Value) -> bool {
    move |t: &Value| {
        let parsed = parse(t);
        predicate(&parsed)
    }
}

pub fn is_borrowed_cool() -> impl for<'a> Fn(&Borrowed<'a>) -> bool {
    |b| true
}

pub fn main() {
    let a = not(is_borrowed_cool());
    let b = borrowed(is_borrowed_cool());
    // I would like this to compile. It doesn't though, and it generates
    // an incorrect diagnostic.
    let c = borrowed(not(is_borrowed_cool()));
}