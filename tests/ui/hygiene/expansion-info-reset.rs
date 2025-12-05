fn main() {
    format_args!({ #[derive(Clone)] struct S; });
    //~^ ERROR format argument must be a string literal
}
