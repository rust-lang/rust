#[unsafe(diagnostic::on_unimplemented(message = "xyz"))] //~ ERROR `diagnostic::on_unimplemented` is not an unsafe attribute
trait Test1 {}

#[unsafe(diagnostic::invalid)]
//~^ ERROR `diagnostic::invalid` is not an unsafe attribute
//~| WARN unknown diagnostic attribute
trait Test2 {}

fn main() { }
