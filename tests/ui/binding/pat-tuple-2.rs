// run-pass
fn tuple() {
    let x = (1,);
    match x {
        (2, ..) => panic!(),
        (..) => ()
    }
}

fn tuple_struct() {
    struct S(u8);

    let x = S(1);
    match x {
        S(2, ..) => panic!(),
        S(..) => ()
    }
}

fn main() {
    tuple();
    tuple_struct();
}
