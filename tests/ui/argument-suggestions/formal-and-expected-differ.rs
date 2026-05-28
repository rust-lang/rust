pub trait Foo {
    type T;
}

impl Foo for i32 {
    type T = f32;
}

pub struct U<T1, T2>(T1, S<T2>)
where
    T1: Foo<T = T2>;

pub struct S<T>(T);

fn main() {
    // The error message here isn't great -- it has to do with the fact that the
    // `expected_inputs_for_expected_output` deduced inputs differs from the inputs
    // that we infer from the constraints of the signature.
    //
    // I am not really sure what the best way of presenting this error message is,
    // since right now it just suggests changing `3u32` <=> `3f32` back and forth.
    let _: U<_, u32> = U(1, S(3u32));
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}
