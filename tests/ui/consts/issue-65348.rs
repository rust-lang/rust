struct Generic<T>(T);

impl<T> Generic<T> {
    const ARRAY: [T; 0] = [];
    const NEWTYPE_ARRAY: Generic<[T; 0]> = Generic([]);
    const ARRAY_FIELD: Generic<(i32, [T; 0])> = Generic((0, []));
}

pub const fn array<T>() ->  &'static T {
    &Generic::<T>::ARRAY[0]
    //~^ ERROR destructor of `[T; 0]` cannot be evaluated at compile-time
}

pub const fn newtype_array<T>() ->  &'static T {
    &Generic::<T>::NEWTYPE_ARRAY.0[0]
    //~^ ERROR destructor of `Generic<[T; 0]>` cannot be evaluated at compile-time
}

pub const fn array_field<T>() ->  &'static T {
    &(Generic::<T>::ARRAY_FIELD.0).1[0]
    //~^ ERROR destructor of `Generic<(i32, [T; 0])>` cannot be evaluated at compile-time
}

fn main() {}
