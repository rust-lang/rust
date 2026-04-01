//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#202. We have
// to make sure we don't constrain ambiguous GAT args when normalizing
// via where bounds or item bounds.

trait Trait {
    type Assoc<U>;
}

fn ret<T: Trait, U>(x: U) -> <T as Trait>::Assoc<U> {
    loop {}
}

fn where_bound<T: Trait<Assoc<u32> = u32>>() {
    let inf = Default::default();
    let x = ret::<T, _>(inf);
    let _: i32 = inf;
}

trait ItemBound {
    type Bound: Trait<Assoc<u32> = u32>;
}
fn item_bound<T: ItemBound>() {
    let inf = Default::default();
    let x = ret::<T::Bound, _>(inf);
    let _: i32 = inf;
}

fn main() {}
