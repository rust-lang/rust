// See #130043 and #130031

fn main() {
    let mut data = [1, 2, 3];
    let mut i = indices(&data);
    data = [4, 5, 6];
    i.next();

    let mut i = enumerated_opaque(&data);
    i.next();

    let mut i = enumerated(&data);
    i.next();

    let mut i = enumerated_lt(&data);
    i.next();

    let mut i = enumerated_arr(data);
    i.next();
}

// No lifetime or type params captured
fn indices<T>(slice: &[T]) -> impl Iterator<Item = usize> + use<> {
    0..slice.len()
}

// `'_` and `T` are captured
fn enumerated_opaque<T>(slice: &[T]) -> impl Iterator + use<> {
    slice.iter().enumerate()
    //~^ ERROR hidden type for `impl Iterator` captures lifetime that does not appear in bounds
}

// `'_` and `T` are captured
fn enumerated_opaque_lt<T>(slice: &[T]) -> impl Iterator + use<'_> {
    //~^ ERROR `impl Trait` must mention all used type parameters in scope in `use<...>`
    slice.iter().enumerate()
}

// `'_` and `T` are captured
fn enumerated<T>(slice: &[T]) -> impl Iterator<Item = (usize, &T)> + use<> {
    //~^ ERROR `impl Trait` must mention all used type parameters in scope in `use<...>`
    slice.iter().enumerate()
}

// `'_` and `T` are captured
fn enumerated_lt<T>(slice: &[T]) -> impl Iterator<Item = (usize, &T)> + use<'_> {
    //~^ ERROR `impl Trait` must mention all used type parameters in scope in `use<...>`
    slice.iter().enumerate()
}

// `T` and `N` are captured
fn enumerated_arr<T, const N: usize>(arr: [T; N]) -> impl Iterator<Item = (usize, T)> + use<> {
    //~^ ERROR `impl Trait` must mention all used type parameters in scope in `use<...>`
    //~| ERROR `impl Trait` must mention all used const parameters in scope in `use<...>`
    <[T; N]>::into_iter(arr).enumerate()
}
