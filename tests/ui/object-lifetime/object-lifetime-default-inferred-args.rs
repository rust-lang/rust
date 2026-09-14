// Ensure that trait object lifetime defaulting properly deals with inferred args `_`.
// RBV which computes the defaults can't tell whether `_` refers to a type or const.
// Therefore it needs to map const params to a dummy trait object lifetime default
// in order to properly align params and args.
//@ check-pass

struct Ty0<'a, Bait: 'static, T: 'a + ?Sized>(Bait, &'a T);

fn check0<'r>(x: Ty0<'r, (), dyn Inner>) {
    // The algorithm can't just avoid mapping const params to defaults and skip const & inferred
    // args when mapping defaults to type args since the inferred arg may be a type.
    // If it did, it would wrongly obtain the default provided by param `Bait` and elaborate
    // `dyn Trait` to `dyn Trait + 'static` (which would fail since `'r` doesn't outlive `'static`).
    let _: Ty0<'r, _, dyn Inner> = x;
}

struct Ty1<'a, const BAIT: usize, T: 'a + ?Sized, Bait: 'static>(&'a T, Bait);

fn check1<'r>(x: Ty1<'r, 0, dyn Inner, ()>) {
    // The algorithm can't just avoid mapping const params to defaults, skip const args and count
    // inferred args as types when mapping defaults to type args since the inferred arg may be a
    // const. If it did, it would wrongly obtain the default provided by param `Bait` and elaborate
    // `dyn Trait` to `dyn Trait + 'static` (which would fail since `'r` doesn't outlive `'static`).
    let _: Ty1<'r, _, dyn Inner, _> = x;
}

trait Inner {}

fn main() {}
