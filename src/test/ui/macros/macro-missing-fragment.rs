macro_rules! m {
    ( $( any_token $field_rust_type )* ) => {}; //~ ERROR missing fragment
}

fn main() {
    m!();
}
