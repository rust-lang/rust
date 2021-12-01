// run-rustfix
// edition:2018

#![allow(unused)]
#![deny(elided_lifetimes_in_paths)]
//~^ NOTE the lint level is defined here

use std::cell::{Ref, RefCell};

struct Foo<'a> {
    x: &'a u32,
}

fn foo(x: &Foo) {
    //~^ ERROR hidden lifetime parameters in types are deprecated
    //~| NOTE expected named lifetime parameter
    //~| HELP consider using the `'_` lifetime
}

fn bar(x: &Foo<'_>) {}

struct Wrapped<'a>(&'a str);

struct WrappedWithBow<'a> {
    gift: &'a str,
}

struct MatchedSet<'a, 'b> {
    one: &'a str,
    another: &'b str,
}

fn wrap_gift(gift: &str) -> Wrapped {
    //~^ ERROR hidden lifetime parameters in types are deprecated
    //~| NOTE expected named lifetime parameter
    //~| HELP consider using the `'_` lifetime
    Wrapped(gift)
}

fn wrap_gift_with_bow(gift: &str) -> WrappedWithBow {
    //~^ ERROR hidden lifetime parameters in types are deprecated
    //~| NOTE expected named lifetime parameter
    //~| HELP consider using the `'_` lifetime
    WrappedWithBow { gift }
}

fn inspect_matched_set(set: MatchedSet) {
    //~^ ERROR hidden lifetime parameters in types are deprecated
    //~| NOTE expected 2 lifetime parameters
    //~| HELP consider using the `'_` lifetime
    println!("{} {}", set.one, set.another);
}

// Verify that the lint does not fire, because the added `'_` wouldn't be resolved correctly.
fn match_sets() -> MatchedSet {
    //~^ ERROR missing lifetime specifiers
    //~| NOTE expected 2 lifetime parameters
    //~| HELP this function's return type contains a borrowed value
    //~| HELP consider using the `'static` lifetime
    MatchedSet { one: "one", another: "another" }
}

macro_rules! autowrapper {
    ($type_name:ident, $fn_name:ident, $lt:lifetime) => {
        struct $type_name<$lt> {
            gift: &$lt str
        }

        fn $fn_name(gift: &str) -> $type_name {
            //~^ ERROR hidden lifetime parameters in types are deprecated
            //~| NOTE expected named lifetime parameter
            //~| HELP consider using the `'_` lifetime
            //~| ERROR hidden lifetime parameters in types are deprecated
            //~| NOTE expected named lifetime parameter
            //~| HELP consider using the `'_` lifetime
            $type_name { gift }
        }
    }
}

autowrapper!(Autowrapped, autowrap_gift, 'a);
//~^ NOTE in this expansion of autowrapper!
//~| NOTE in this expansion of autowrapper!

// Verify that rustfix does not try to apply the fix twice.
autowrapper!(AutowrappedAgain, autowrap_gift_again, 'a);
//~^ NOTE in this expansion of autowrapper!
//~| NOTE in this expansion of autowrapper!

macro_rules! anytuple_ref_ty {
    ($($types:ty),*) => {
        Ref<($($types),*)>
        //~^ ERROR hidden lifetime parameters in types are deprecated
        //~| NOTE expected named lifetime parameter
        //~| HELP consider using the `'_` lifetime
    }
}

#[allow(elided_lifetimes_in_paths)]
mod blah {
    struct Thing<'a>(&'a i32);
    struct Bar<T>(T);

    fn foo(b: Bar<Thing>) {}
}

fn main() {
    let honesty = RefCell::new((4, 'e'));
    let loyalty: Ref<(u32, char)> = honesty.borrow();
    //~^ ERROR hidden lifetime parameters in types are deprecated
    //~| NOTE expected named lifetime parameter
    //~| HELP consider using the `'_` lifetime
    let generosity = Ref::map(loyalty, |t| &t.0);

    let laughter = RefCell::new((true, "magic"));
    let yellow: anytuple_ref_ty!(bool, &str) = laughter.borrow();
    //~^ NOTE in this expansion of anytuple_ref_ty!
    //~| NOTE in this expansion of anytuple_ref_ty!
}
