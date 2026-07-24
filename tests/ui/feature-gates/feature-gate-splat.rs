#[rustfmt::skip]
fn tuple_args(
    #[arg_splat] //~ ERROR the `#[arg_splat]` attribute is an experimental feature
    (a, b, c): (u32, i8, char),
) {
}

fn main() {}
