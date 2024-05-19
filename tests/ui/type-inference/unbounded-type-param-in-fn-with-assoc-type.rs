#[allow(invalid_type_param_default)]

fn foo<T, U = u64>() -> (T, U) {
    panic!()
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
