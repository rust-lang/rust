//@ compile-flags: -Znext-solver
//@ check-pass

// Previously we didn't rerun the goal with doubled recursion limit if the goal contained ty vars.
// This is to avoid futile evaluation. However, sometimes the type inference progress relies on
// successful trait solving response. If we don't evaluate with higher recursion limit, the type
// inference would fail eventually.
//
// See the `recursion_depth_exceeding_limit` FCW for why we need the doubled recursion limit.

// Setting it to 12 would make it compile.
#![recursion_limit = "6"]


//~^^^^^^^^^^^^^^ WARN: overflow evaluating the requirement `(): Trait<i32>` [recursion_depth_exceeding_limit]
//~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

trait Trait<T> {}

struct W1<T>(T);
struct W2<T>(T);
struct W3<T>(T);
struct W4<T>(T);
struct W5<T>(T);
struct W6<T>(T);
struct W7<T>(T);


impl<T> Trait<T> for ()
    where
    W1<T>: Trait<T>,
{}

impl<T> Trait<T> for W1<T>
where
    W2<T>: Trait<T>,
{}

impl<T> Trait<T> for W2<T>
where
    W3<T>: Trait<T>,
{}

impl<T> Trait<T> for W3<T>
where
    W4<T>: Trait<T>,
{}

impl<T> Trait<T> for W4<T>
where
    W5<T>: Trait<T>,
{}

impl<T> Trait<T> for W5<T>
where
    W6<T>: Trait<T>,
{}

impl<T> Trait<T> for W6<T>
where
    W7<T>: Trait<T>,
{}

impl Trait<i32> for W7<i32> {}

fn foo<T>()
    where
        (): Trait<T>,
{
}

fn main() {
    foo(); // register a `(): Trait<?t>` obligation
    //~^ WARN: overflow evaluating the requirement `(): Trait<_>` [recursion_depth_exceeding_limit]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

}
