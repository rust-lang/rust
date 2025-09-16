#![feature(rustc_attrs, repr_simd)]

//@ build-fail

#[repr(simd, packed)]
#[rustc_simd_monomorphize_lane_limit = "4"]
struct V<T, const N: usize>([T; N]);

struct Container<T, const N: usize>(V<T, N>);

fn main() {
    power_of_two();
    non_power_of_two();
}

fn fn_args<const N: usize>(v: Option<Container<i32, N>>) {
    //~^ ERROR exceeding the limit 4
    //~^^ ERROR exceeding the limit 4
    unimplemented!()
}

fn fn_ret<const N: usize>() -> Container<i32, N> {
    //~^ ERROR exceeding the limit 4
    //~^^ ERROR exceeding the limit 4
    unimplemented!()
}

fn power_of_two() {
    const LANES: usize = 8;

    let _a: V<i32, LANES> = V([0; LANES]); //~ ERROR exceeding the limit 4

    let _b = fn_args::<LANES>(None);

    let _c = fn_ret::<LANES>();
}

fn non_power_of_two() {
    const LANES: usize = 6;

    let _a: V<i32, LANES> = V([0; LANES]); //~ ERROR exceeding the limit 4

    let _b = fn_args::<LANES>(None);

    let _c = fn_ret::<LANES>();
}
