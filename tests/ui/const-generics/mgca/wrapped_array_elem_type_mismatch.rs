#![expect(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args)]

struct ArrWrap<const N: [u8; 1]>;

fn main() {
    let _: ArrWrap<{ [1_u8] }> = ArrWrap::<{ [1_u16] }>;
    //~^ ERROR the literal is not of type `u8`
}
