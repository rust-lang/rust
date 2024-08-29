#[unsafe(diagnostic::on_unimplemented( //~ ERROR: is not an unsafe attribute
    message = "testing",
))]
trait Foo {}

fn main() {}
