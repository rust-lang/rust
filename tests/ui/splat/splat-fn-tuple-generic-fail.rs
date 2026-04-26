//! Test failing use of `#[splat]` on tuple trait arguments of generic functions.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn splat_generic_tuple<T: std::marker::Tuple>(#[splat] _t: T) {}

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?

    // Calling with un-splatted arguments might look like it works, but the actual generic type is
    // a tuple inside another tuple. Aren't generics great?
    splat_generic_tuple((1, 2));
    splat_generic_tuple((1u32, 2i8));

    // FIXME(splat): Make the splat generic handling code handle tuples inside tuples
    // (if we want to support tupled calls)
    splat_generic_tuple::<(((u32, i8)))>((1, 2)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided
    splat_generic_tuple::<(((u32, i8)))>((1u32, 2i8)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided

    splat_generic_tuple::<((u32, i8))>((1, 2)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided
    splat_generic_tuple::<((u32, i8))>((1u32, 2i8)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided

    splat_generic_tuple::<(u32, i8)>((1, 2)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided
    splat_generic_tuple::<(u32, i8)>((1u32, 2i8)); //~ ERROR this splatted function takes 2 arguments, but 1 was provided
}
