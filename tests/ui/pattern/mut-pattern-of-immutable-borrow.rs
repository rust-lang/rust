struct S {
    field: Option<String>,
}

fn a(arg: &mut S) {
    match arg.field { //~ ERROR cannot move out of `arg.field`
        Some(s) => s.push('a'), //~ ERROR cannot borrow `s` as mutable
        None => {}
    }
}
fn b(arg: &mut S) {
    match &arg.field { //~ ERROR cannot move out of a shared reference
        Some(mut s) => s.push('a'),
        None => {}
    }
}
fn c(arg: &mut S) {
    match &arg.field {
        Some(ref mut s) => s.push('a'), //~ ERROR cannot borrow data in a `&` reference as mutable
        None => {}
    }
}

fn main() {
    let mut s = S {
        field: Some("a".to_owned()),
    };
    a(&mut s);
    b(&mut s);
    c(&mut s);
}
