enum E {
    A(u8, u8),
}

fn main() {
    let e = E::A(2, 3);
    match e {
        E::A(x @ ..) => {  //~ ERROR `x @` is not allowed in a tuple
            x //~ ERROR cannot find value `x` in this scope
        }
    };
}
