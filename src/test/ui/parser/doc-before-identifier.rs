// compile-flags: -Z continue-parse-after-error

fn /// document
foo() {}
//~^^ ERROR expected identifier, found doc comment `/// document`

fn main() {
    foo();
}
