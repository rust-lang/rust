//@ edition:2024

fn temp() -> String {
    String::from("Hello")
}

#[derive(Debug)]
struct X<'a>(&'a String);

fn main() {
    let a = &temp();
    let b = Some(&temp());
    let c = Option::Some::<&String>(&temp());
    use Option::Some as S;
    let d = S(&temp());
    let e = X(&temp());
    let f = Some(Ok::<_, ()>(std::borrow::Cow::Borrowed(if true {
        &temp()
    } else {
        panic!()
    })));
    let some = Some; // Turn the ctor into a regular function.
    let g = some(&temp()); //~ERROR temporary value dropped while borrowe
    println!("{a:?} {b:?} {c:?} {d:?} {e:?} {f:?} {g:?}");
}
