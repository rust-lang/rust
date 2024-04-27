//@ check-pass

enum Sexpression {
    Num(()),
    Cons(&'static mut Sexpression)
}

fn causes_error_in_ast(mut l: &mut Sexpression) {
    loop { match l {
        &mut Sexpression::Num(ref mut n) => {},
        &mut Sexpression::Cons(ref mut expr) => {
            l = &mut **expr;
        }
    }}
}


fn main() {
    causes_error_in_ast(&mut Sexpression::Num(()));
}
