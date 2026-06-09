use std::format_args_nl; //~ ERROR `format_args_nl` is only for internal language use

fn main() {
    format_args_nl!(""); //~ ERROR `format_args_nl` is only for internal language use
}
