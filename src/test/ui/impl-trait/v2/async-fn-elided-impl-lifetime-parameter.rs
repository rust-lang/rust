// Check that `async fn` inside of an impl with `'_`
// in the header compiles correctly.
//
// Regression test for #63500.
//
// check-pass
// edition:2018

#![feature(return_position_impl_trait_v2)]

struct Foo<'a>(&'a u8);

impl Foo<'_> { // impl Foo<'(fresh:NodeId0)>
// ^ extra_lifetime_parameters: [fresh:NodeId0]
//
// we insert these into `node_id_to_def_id_override`

    async fn bar() {}
    // but HERE, when you have an async function, you create a node-id for it (opaque_ty_id)
    //
    // and we attach (using `extra_lifetime_parameters`) a list of *all the lifetimes that are in scope from the impl and the fn*
    //
    // 
}

// for<> fn(&'(fresh:NodeId0) u32)
// ^
// extra_lifetime_parameters: [fresh:NodeId0] instantiate def-ids for them and add them into the node_id_to_def_id_override
//                            with the node id NodeId0
//
// then later when we process `fresh:NodeId0` we look up in `node_id_to_def_id_override` and we find it

fn main() {}
