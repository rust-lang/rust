// run-pass

#![allow(warnings)]
#![feature(generators)]

use std::fmt::Debug;

fn any_lifetime<'a>() -> &'a u32 { &5 }

fn static_lifetime() -> &'static u32 { &5 }

fn any_lifetime_as_static_impl_trait() -> impl Debug {
    any_lifetime()
}

fn lifetimes_as_static_impl_trait() -> impl Debug {
    static_lifetime()
}

fn no_params_or_lifetimes_is_static() -> impl Debug + 'static {
    lifetimes_as_static_impl_trait()
}

fn static_input_type_is_static<T: Debug + 'static>(x: T) -> impl Debug + 'static { x }

fn type_outlives_reference_lifetime<'a, T: Debug>(x: &'a T) -> impl Debug + 'a { x }
fn type_outlives_reference_lifetime_elided<T: Debug>(x: &T) -> impl Debug + '_ { x }

trait SingleRegionTrait<'a> {}
impl<'a> SingleRegionTrait<'a> for u32 {}
impl<'a> SingleRegionTrait<'a> for &'a u32 {}
struct SingleRegionStruct<'a>(&'a u32);

fn simple_type_hrtb<'b>() -> impl for<'a> SingleRegionTrait<'a> { 5 }
fn elision_single_region_trait(x: &u32) -> impl SingleRegionTrait { x }
fn elision_single_region_struct(x: SingleRegionStruct) -> impl Into<SingleRegionStruct> { x }

fn closure_hrtb() -> impl for<'a> Fn(&'a u32) { |_| () }
fn closure_hr_elided() -> impl Fn(&u32) { |_| () }
fn closure_hr_elided_return() -> impl Fn(&u32) -> &u32 { |x| x }
fn closure_pass_through_elided_return(x: impl Fn(&u32) -> &u32) -> impl Fn(&u32) -> &u32 { x }
fn closure_pass_through_reference_elided(x: &impl Fn(&u32) -> &u32) -> &impl Fn(&u32) -> &u32 { x }

fn nested_lifetime<'a>(input: &'a str)
    -> impl Iterator<Item = impl Iterator<Item = i32> + 'a> + 'a
{
    input.lines().map(|line| {
        line.split_whitespace().map(|cell| cell.parse().unwrap())
    })
}

fn pass_through_elision(x: &u32) -> impl Into<&u32> { x }
fn pass_through_elision_with_fn_ptr(x: &fn(&u32) -> &u32) -> impl Into<&fn(&u32) -> &u32> { x }

fn pass_through_elision_with_fn_path<T: Fn(&u32) -> &u32>(
    x: &T
) -> &impl Fn(&u32) -> &u32 { x }

fn foo(x: &impl Debug) -> &impl Debug { x }
fn foo_explicit_lifetime<'a>(x: &'a impl Debug) -> &'a impl Debug { x }
fn foo_explicit_arg<T: Debug>(x: &T) -> &impl Debug { x }

fn mixed_lifetimes<'a>() -> impl for<'b> Fn(&'b &'a u32) { |_| () }
fn mixed_as_static() -> impl Fn(&'static &'static u32) { mixed_lifetimes() }

trait MultiRegionTrait<'a, 'b>: Debug {}

#[derive(Debug)]
struct MultiRegionStruct<'a, 'b>(&'a u32, &'b u32);
impl<'a, 'b> MultiRegionTrait<'a, 'b> for MultiRegionStruct<'a, 'b> {}

#[derive(Debug)]
struct NoRegionStruct;
impl<'a, 'b> MultiRegionTrait<'a, 'b> for NoRegionStruct {}

fn finds_least_region<'a: 'b, 'b>(x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> {
    MultiRegionStruct(x, y)
}

fn finds_explicit_bound<'a: 'b, 'b>
    (x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> + 'b
{
    MultiRegionStruct(x, y)
}

fn finds_explicit_bound_even_without_least_region<'a, 'b>
    (x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> + 'b
{
    NoRegionStruct
}

/* FIXME: `impl Trait<'a> + 'b` should live as long as 'b, even if 'b outlives 'a
fn outlives_bounds_even_with_contained_regions<'a, 'b>
    (x: &'a u32, y: &'b u32) -> impl Debug + 'b
{
    finds_explicit_bound_even_without_least_region(x, y)
}
*/

fn unnamed_lifetimes_arent_contained_in_impl_trait_and_will_unify<'a, 'b>
    (x: &'a u32, y: &'b u32) -> impl Debug
{
    fn deref<'lt>(x: &'lt u32) -> impl Debug { *x }

    if true { deref(x) } else { deref(y) }
}

fn can_add_region_bound_to_static_type<'a, 'b>(_: &'a u32) -> impl Debug + 'a { 5 }

struct MyVec(Vec<Vec<u8>>);

impl<'unnecessary_lifetime> MyVec {
    fn iter_doesnt_capture_unnecessary_lifetime<'s>(&'s self) -> impl Iterator<Item = &'s u8> {
        self.0.iter().flat_map(|inner_vec| inner_vec.iter())
    }

    fn generator_doesnt_capture_unnecessary_lifetime<'s: 's>() -> impl Sized {
        || yield
    }
}


fn main() {}
