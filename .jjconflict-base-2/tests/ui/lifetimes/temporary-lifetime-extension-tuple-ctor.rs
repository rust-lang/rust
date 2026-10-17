//@ reference: destructors.scope.lifetime-extension.exprs

fn temp() -> String {
    String::from("Hello")
}

#[derive(Debug)]
struct X<'a>(&'a String);

trait T<'a> {
    const A: X<'a>;
    const B: X<'a>;
}

impl<'a> T<'a> for X<'a> {
    // Check both Self() and X() syntax:
    const A: X<'a> = Self(&String::new());
    const B: X<'a> = X(&String::new());
}

fn main() {
    let a = &temp();
    let b = Some(&temp());
    let c = Option::Some::<&String>(&temp());
    use std::option::Option::Some as S;
    let d = S(&temp());
    let e = X(&temp());
    let f = Some(Ok::<_, ()>(std::borrow::Cow::Borrowed(if true {
        &temp()
    } else {
        panic!()
    })));
    let some = Some; // Turn the ctor into a regular function.
    let g = some(&temp()); //~ERROR temporary value dropped while borrowed
    println!("{a:?} {b:?} {c:?} {d:?} {e:?} {f:?} {g:?}");
}
