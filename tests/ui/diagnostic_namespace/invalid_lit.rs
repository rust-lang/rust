#[diagnostic::on_unimplemented = 1 + 1]
//~^ ERROR attribute value must be a literal
#[diagnostic::not_exist = 1 + 1]
//~^ ERROR attribute value must be a literal
//~| WARN unknown diagnostic attribute
fn main() {

}
