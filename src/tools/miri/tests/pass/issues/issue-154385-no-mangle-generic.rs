fn main() {
    foo(1234);
}

#[allow(no_mangle_generic_items)]
#[unsafe(no_mangle)]
fn foo<T: std::fmt::Debug>(value: T) {
    println!("{value:?}");
}
