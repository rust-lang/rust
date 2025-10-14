#[unsafe(diagnostic::on_unimplemented( //~ ERROR: unnecessary `unsafe`
    message = "testing",
))]
trait Foo {}

fn main() {}
