#[rustfmt::skip]
fn tuple_args(
    #[rustc_splat] //~ ERROR the `rustc_splat` attribute is an experimental feature
    (a, b, c): (u32, i8, char),
) {
}

fn main() {}
