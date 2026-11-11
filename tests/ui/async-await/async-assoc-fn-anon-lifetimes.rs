//@ check-pass
// Check that the anonymous lifetimes used here aren't considered to shadow one
// another. Note that `async fn` is different to `fn` here because the lifetimes
// are numbered by HIR lowering, rather than lifetime resolution.

//@ edition:2018

#![allow(non_local_definitions)]

struct A<'a, 'b>(&'a &'b i32);
struct B<'a>(&'a i32);

impl A<'_, '_> {
    async fn assoc(x: &u32, y: B<'_>) {
        async fn nested(x: &u32, y: A<'_, '_>) {}
    }

    async fn assoc2(x: &u32, y: A<'_, '_>) {
        impl A<'_, '_> {
            async fn nested_assoc(x: &u32, y: B<'_>) {}
        }
    }
}

fn main() {}
