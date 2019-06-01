fn /// document
foo() {}
//~^^ ERROR expected identifier, found doc comment `/// document`

fn main() {
    foo();
}
