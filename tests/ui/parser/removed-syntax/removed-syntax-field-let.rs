struct S {
    let foo: (),
    //~^  ERROR expected identifier, found keyword `let`
}

fn main() {}
