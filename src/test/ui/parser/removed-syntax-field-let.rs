struct S {
    let foo: (),
    //~^  ERROR expected identifier, found keyword `let`
    //~^^ ERROR expected `:`, found `foo`
}

fn main() {}
