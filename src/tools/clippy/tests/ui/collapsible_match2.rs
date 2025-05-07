#![warn(clippy::collapsible_match)]
#![allow(
    clippy::needless_return,
    clippy::no_effect,
    clippy::single_match,
    clippy::needless_borrow
)]

fn lint_cases(opt_opt: Option<Option<u32>>, res_opt: Result<Option<u32>, String>) {
    // if guards on outer match
    {
        match res_opt {
            Ok(val) if make() => match val {
                //~^ collapsible_match
                Some(n) => foo(n),
                _ => return,
            },
            _ => return,
        }
        match res_opt {
            Ok(val) => match val {
                //~^ collapsible_match
                Some(n) => foo(n),
                _ => return,
            },
            _ if make() => return,
            _ => return,
        }
    }

    // macro
    {
        macro_rules! mac {
            ($outer:expr => $pat:pat, $e:expr => $inner_pat:pat, $then:expr) => {
                match $outer {
                    $pat => match $e {
                        //~^ collapsible_match
                        $inner_pat => $then,
                        _ => return,
                    },
                    _ => return,
                }
            };
        }
        // Lint this since the patterns are not defined by the macro.
        // Allows the lint to work on if_chain! for example.
        // Fixing the lint requires knowledge of the specific macro, but we optimistically assume that
        // there is still a better way to write this.
        mac!(res_opt => Ok(val), val => Some(n), foo(n));
    }

    // deref reference value
    match Some(&[1]) {
        Some(s) => match *s {
            //~^ collapsible_match
            [n] => foo(n),
            _ => (),
        },
        _ => (),
    }

    // ref pattern and deref
    match Some(&[1]) {
        Some(ref s) => match s {
            //~^ collapsible_match
            [n] => foo(n),
            _ => (),
        },
        _ => (),
    }
}

fn no_lint() {
    // deref inner value (cannot pattern match with Vec)
    match Some(vec![1]) {
        Some(s) => match *s {
            [n] => foo(n),
            _ => (),
        },
        _ => (),
    }
}

fn make<T>() -> T {
    unimplemented!()
}

fn foo<T, U>(t: T) -> U {
    unimplemented!()
}

fn main() {}
