#[rustfmt::skip]
fn tuple_args(
    #[splat] //~ ERROR the `#[splat]` attribute is an experimental feature
    (a, b, c): (u32, i8, char),
) {
}

fn main() {}
