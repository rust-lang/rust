// compile-flags: -Z parse-only -Z continue-parse-after-error

struct s {
    let foo: (),
    //~^  ERROR expected identifier, found keyword `let`
    //~^^ ERROR expected `:`, found `foo`
}
