#[unsafe(diagnostic::on_unimplemented(message = "xyz"))] //~ ERROR `diagnostic::on_unimplemented` is not an unsafe attribute
trait Test1 {}

#[unsafe(diagnostic::invalid)] //~ WARN unknown diagnostic attribute
trait Test2 {}

fn main() { }
