// Targeted tests for the higher-ranked subtyping code.

#![feature(rustc_attrs)]
#![allow(dead_code)]

// revisions: bound_a_vs_bound_a
// revisions: bound_a_vs_bound_b
// revisions: bound_inv_a_vs_bound_inv_b
// revisions: bound_co_a_vs_bound_co_b
// revisions: bound_a_vs_free_x
// revisions: free_x_vs_free_x
// revisions: free_x_vs_free_y
// revisions: free_inv_x_vs_free_inv_y
// revisions: bound_a_b_vs_bound_a
// revisions: bound_co_a_b_vs_bound_co_a
// revisions: bound_contra_a_contra_b_ret_co_a
// revisions: bound_co_a_co_b_ret_contra_a
// revisions: bound_inv_a_b_vs_bound_inv_a
// revisions: bound_a_b_ret_a_vs_bound_a_ret_a

fn gimme<T>(_: Option<T>) { }

struct Inv<'a> { x: *mut &'a u32 }

struct Co<'a> { x: fn(&'a u32) }

struct Contra<'a> { x: &'a u32 }

macro_rules! check {
    ($rev:ident: ($t1:ty, $t2:ty)) => {
        #[cfg($rev)]
        fn subtype<'x,'y:'x,'z:'y>() {
            gimme::<$t2>(None::<$t1>);
            //[free_inv_x_vs_free_inv_y]~^ ERROR
        }

        #[cfg($rev)]
        fn supertype<'x,'y:'x,'z:'y>() {
            gimme::<$t1>(None::<$t2>);
            //[bound_a_vs_free_x]~^ ERROR
            //[free_x_vs_free_y]~^^ ERROR
            //[bound_inv_a_b_vs_bound_inv_a]~^^^ ERROR
            //[bound_a_b_ret_a_vs_bound_a_ret_a]~^^^^ ERROR
            //[free_inv_x_vs_free_inv_y]~^^^^^ ERROR
            //[bound_a_b_vs_bound_a]~^^^^^^ ERROR mismatched types
            //[bound_co_a_co_b_ret_contra_a]~^^^^^^^ ERROR
            //[bound_contra_a_contra_b_ret_co_a]~^^^^^^^^ ERROR
            //[bound_co_a_b_vs_bound_co_a]~^^^^^^^^^ ERROR
        }
    }
}

// If both have bound regions, they are equivalent, regardless of
// variant.
check! { bound_a_vs_bound_a: (for<'a> fn(&'a u32),
                              for<'a> fn(&'a u32)) }
check! { bound_a_vs_bound_b: (for<'a> fn(&'a u32),
                              for<'b> fn(&'b u32)) }
check! { bound_inv_a_vs_bound_inv_b: (for<'a> fn(Inv<'a>),
                                      for<'b> fn(Inv<'b>)) }
check! { bound_co_a_vs_bound_co_b: (for<'a> fn(Co<'a>),
                                    for<'b> fn(Co<'b>)) }

// Bound is a subtype of free.
check! { bound_a_vs_free_x: (for<'a> fn(&'a u32),
                             fn(&'x u32)) }

// Two free regions are relatable if subtyping holds.
check! { free_x_vs_free_x: (fn(&'x u32),
                            fn(&'x u32)) }
check! { free_x_vs_free_y: (fn(&'x u32),
                            fn(&'y u32)) }
check! { free_inv_x_vs_free_inv_y: (fn(Inv<'x>),
                                    fn(Inv<'y>)) }

// Somewhat surprisingly, a fn taking two distinct bound lifetimes and
// a fn taking one bound lifetime can be interchangeable, but only if
// we are co- or contra-variant with respect to both lifetimes.
//
// The reason is:
// - if we are covariant, then 'a and 'b can be set to the call-site
//   intersection;
// - if we are contravariant, then 'a can be inferred to 'static.
check! { bound_a_b_vs_bound_a: (for<'a,'b> fn(&'a u32, &'b u32),
                                for<'a>    fn(&'a u32, &'a u32)) }
check! { bound_co_a_b_vs_bound_co_a: (for<'a,'b> fn(Co<'a>, Co<'b>),
                                      for<'a>    fn(Co<'a>, Co<'a>)) }
check! { bound_contra_a_contra_b_ret_co_a: (for<'a,'b> fn(Contra<'a>, Contra<'b>) -> Co<'a>,
                                            for<'a>    fn(Contra<'a>, Contra<'a>) -> Co<'a>) }
check! { bound_co_a_co_b_ret_contra_a: (for<'a,'b> fn(Co<'a>, Co<'b>) -> Contra<'a>,
                                        for<'a>    fn(Co<'a>, Co<'a>) -> Contra<'a>) }

// If we make those lifetimes invariant, then the two types are not interchangeable.
check! { bound_inv_a_b_vs_bound_inv_a: (for<'a,'b> fn(Inv<'a>, Inv<'b>),
                                        for<'a>    fn(Inv<'a>, Inv<'a>)) }
check! { bound_a_b_ret_a_vs_bound_a_ret_a: (for<'a,'b> fn(&'a u32, &'b u32) -> &'a u32,
                                            for<'a>    fn(&'a u32, &'a u32) -> &'a u32) }

#[rustc_error]
fn main() {
//[bound_a_vs_bound_a]~^ ERROR compilation successful
//[bound_a_vs_bound_b]~^^ ERROR compilation successful
//[bound_inv_a_vs_bound_inv_b]~^^^ ERROR compilation successful
//[bound_co_a_vs_bound_co_b]~^^^^ ERROR compilation successful
//[free_x_vs_free_x]~^^^^^ ERROR compilation successful
}
