// compile-flags: -Z continue-parse-after-error
fn /// document
foo() {}
//~^^ ERROR expected identifier, found `/// document`

fn main() {
    foo();
}
