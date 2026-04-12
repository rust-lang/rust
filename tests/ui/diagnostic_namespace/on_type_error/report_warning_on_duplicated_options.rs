#[diagnostic::on_type_error(message = "one", message = "two")]
struct S<T>(T);

fn main() {}
