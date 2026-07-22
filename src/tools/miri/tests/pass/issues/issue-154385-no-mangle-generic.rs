fn main() {
    generic_type(123);
    generic_const::<456>();
    generic_lifetime(&789);
}

#[allow(no_mangle_generic_items)]
#[unsafe(no_mangle)]
fn generic_type<T: std::fmt::Debug>(value: T) {
    println!("{value:?}");
}

#[expect(no_mangle_generic_items)]
#[unsafe(no_mangle)]
fn generic_const<const N: usize>() {
    println!("{N}");
}

#[unsafe(no_mangle)]
fn generic_lifetime<'a>(x: &'a i32) {
    println!("{x}");
}
