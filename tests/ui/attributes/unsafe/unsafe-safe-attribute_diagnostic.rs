#[unsafe(diagnostic::on_unimplemented( //~ ERROR: `diagnostic::on_unimplemented` is not an unsafe attribute
    message = "testing",
))]
trait Foo {}

fn main() {}
