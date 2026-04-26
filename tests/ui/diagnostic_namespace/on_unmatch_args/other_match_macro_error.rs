#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(
    message = "invalid route method",
    note = "this macro expects a action, like `{This}!(get \"/hello\")`"
)]
macro_rules! route {
    (get $path:literal) => {};
}

fn main() {
    route!(post "/");
    //~^ ERROR invalid route method
}
