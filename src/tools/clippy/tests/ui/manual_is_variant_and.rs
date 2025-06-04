//@aux-build:option_helpers.rs
#![warn(clippy::manual_is_variant_and)]

#[macro_use]
extern crate option_helpers;

struct Foo<T>(T);

impl<T> Foo<T> {
    fn map<F: FnMut(T) -> bool>(self, mut f: F) -> Option<bool> {
        Some(f(self.0))
    }
}

fn foo() -> Option<bool> {
    Some(true)
}

macro_rules! some_true {
    () => {
        Some(true)
    };
}
macro_rules! some_false {
    () => {
        Some(false)
    };
}

macro_rules! mac {
    (some $e:expr) => {
        Some($e)
    };
    (some_map $e:expr) => {
        Some($e).map(|x| x % 2 == 0)
    };
    (map $e:expr) => {
        $e.map(|x| x % 2 == 0)
    };
    (eq $a:expr, $b:expr) => {
        $a == $b
    };
}

#[rustfmt::skip]
fn option_methods() {
    let opt = Some(1);

    // Check for `option.map(_).unwrap_or_default()` use.
    // Single line case.
    let _ = opt.map(|x| x > 1)
    //~^ manual_is_variant_and
        // Should lint even though this call is on a separate line.
        .unwrap_or_default();
    // Multi-line cases.
    let _ = opt.map(|x| {
    //~^ manual_is_variant_and
        x > 1
    }
    ).unwrap_or_default();
    let _ = opt.map(|x| x > 1).unwrap_or_default();
    //~^ manual_is_variant_and
    let _ = opt
        .map(|x| x > 1)
        //~^ manual_is_variant_and
        .unwrap_or_default();

    let _ = Some(2).map(|x| x % 2 == 0) == Some(true);
    //~^ manual_is_variant_and
    let _ = Some(2).map(|x| x % 2 == 0) != Some(true);
    //~^ manual_is_variant_and
    let _ = Some(2).map(|x| x % 2 == 0) == some_true!();
    //~^ manual_is_variant_and
    let _ = Some(2).map(|x| x % 2 == 0) != some_false!();
    //~^ manual_is_variant_and

    // won't fix because the return type of the closure is not `bool`
    let _ = opt.map(|x| x + 1).unwrap_or_default();

    let opt2 = Some('a');
    let _ = opt2.map(char::is_alphanumeric).unwrap_or_default(); // should lint
    //~^ manual_is_variant_and
    let _ = opt_map!(opt2, |x| x == 'a').unwrap_or_default(); // should not lint

    // Should not lint.
    let _ = Foo::<u32>(0).map(|x| x % 2 == 0) == Some(true);
    let _ = Some(2).map(|x| x % 2 == 0) != foo();
    let _ = mac!(eq Some(2).map(|x| x % 2 == 0), Some(true));
    let _ = mac!(some 2).map(|x| x % 2 == 0) == Some(true);
    let _ = mac!(some_map 2) == Some(true);
    let _ = mac!(map Some(2)) == Some(true);
}

#[rustfmt::skip]
fn result_methods() {
    let res: Result<i32, ()> = Ok(1);

    // multi line cases
    let _ = res.map(|x| {
    //~^ manual_is_variant_and
        x > 1
    }
    ).unwrap_or_default();
    let _ = res.map(|x| x > 1)
    //~^ manual_is_variant_and
        .unwrap_or_default();

    let _ = Ok::<usize, ()>(2).map(|x| x % 2 == 0) == Ok(true);
    //~^ manual_is_variant_and
    let _ = Ok::<usize, ()>(2).map(|x| x % 2 == 0) != Ok(true);
    //~^ manual_is_variant_and
    let _ = Ok::<usize, ()>(2).map(|x| x % 2 == 0) != Ok(true);
    //~^ manual_is_variant_and

    // won't fix because the return type of the closure is not `bool`
    let _ = res.map(|x| x + 1).unwrap_or_default();

    let res2: Result<char, ()> = Ok('a');
    let _ = res2.map(char::is_alphanumeric).unwrap_or_default(); // should lint
    //~^ manual_is_variant_and
    let _ = opt_map!(res2, |x| x == 'a').unwrap_or_default(); // should not lint
}

fn main() {
    option_methods();
    result_methods();
}
