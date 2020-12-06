#![warn(clippy::collapsible_match)]
#![allow(clippy::needless_return, clippy::no_effect, clippy::single_match)]

fn lint_cases(opt_opt: Option<Option<u32>>, res_opt: Result<Option<u32>, String>) {
    // if guards on outer match
    {
        match res_opt {
            Ok(val) if make() => match val {
                Some(n) => foo(n),
                _ => return,
            },
            _ => return,
        }
        match res_opt {
            Ok(val) => match val {
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
}

fn make<T>() -> T {
    unimplemented!()
}

fn foo<T, U>(t: T) -> U {
    unimplemented!()
}

fn main() {}
