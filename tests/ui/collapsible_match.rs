#![warn(clippy::collapsible_match)]
#![allow(clippy::needless_return, clippy::no_effect, clippy::single_match)]

fn lint_cases(opt_opt: Option<Option<u32>>, res_opt: Result<Option<u32>, String>) {
    // match without block
    match res_opt {
        Ok(val) => match val {
            Some(n) => foo(n),
            _ => return,
        },
        _ => return,
    }

    // match with block
    match res_opt {
        Ok(val) => match val {
            Some(n) => foo(n),
            _ => return,
        },
        _ => return,
    }

    // if let, if let
    if let Ok(val) = res_opt {
        if let Some(n) = val {
            take(n);
        }
    }

    // if let else, if let else
    if let Ok(val) = res_opt {
        if let Some(n) = val {
            take(n);
        } else {
            return;
        }
    } else {
        return;
    }

    // if let, match
    if let Ok(val) = res_opt {
        match val {
            Some(n) => foo(n),
            _ => (),
        }
    }

    // match, if let
    match res_opt {
        Ok(val) => {
            if let Some(n) = val {
                take(n);
            }
        },
        _ => {},
    }

    // if let else, match
    if let Ok(val) = res_opt {
        match val {
            Some(n) => foo(n),
            _ => return,
        }
    } else {
        return;
    }

    // match, if let else
    match res_opt {
        Ok(val) => {
            if let Some(n) = val {
                take(n);
            } else {
                return;
            }
        },
        _ => return,
    }

    // None in inner match same as outer wild branch
    match res_opt {
        Ok(val) => match val {
            Some(n) => foo(n),
            None => return,
        },
        _ => return,
    }

    // None in outer match same as inner wild branch
    match opt_opt {
        Some(val) => match val {
            Some(n) => foo(n),
            _ => return,
        },
        None => return,
    }
}

fn negative_cases(res_opt: Result<Option<u32>, String>, res_res: Result<Result<u32, String>, String>) {
    // no wild pattern in outer match
    match res_opt {
        Ok(val) => match val {
            Some(n) => foo(n),
            _ => return,
        },
        Err(_) => return,
    }

    // inner branch is not wild or None
    match res_res {
        Ok(val) => match val {
            Ok(n) => foo(n),
            Err(_) => return,
        },
        _ => return,
    }

    // statement before inner match
    match res_opt {
        Ok(val) => {
            "hi buddy";
            match val {
                Some(n) => foo(n),
                _ => return,
            }
        },
        _ => return,
    }

    // statement after inner match
    match res_opt {
        Ok(val) => {
            match val {
                Some(n) => foo(n),
                _ => return,
            }
            "hi buddy";
        },
        _ => return,
    }

    // wild branches do not match
    match res_opt {
        Ok(val) => match val {
            Some(n) => foo(n),
            _ => {
                "sup";
                return;
            },
        },
        _ => return,
    }

    // binding used in if guard
    match res_opt {
        Ok(val) if val.is_some() => match val {
            Some(n) => foo(n),
            _ => return,
        },
        _ => return,
    }

    // binding used in inner match body
    match res_opt {
        Ok(val) => match val {
            Some(_) => take(val),
            _ => return,
        },
        _ => return,
    }

    // if guard on inner match
    {
        match res_opt {
            Ok(val) => match val {
                Some(n) if make() => foo(n),
                _ => return,
            },
            _ => return,
        }
        match res_opt {
            Ok(val) => match val {
                _ => make(),
                _ if make() => return,
            },
            _ => return,
        }
    }

    // differing macro contexts
    {
        macro_rules! mac {
            ($val:ident) => {
                match $val {
                    Some(n) => foo(n),
                    _ => return,
                }
            };
        }
        match res_opt {
            Ok(val) => mac!(val),
            _ => return,
        }
    }

    // OR pattern
    enum E<T> {
        A(T),
        B(T),
        C(T),
    };
    match make::<E<Option<u32>>>() {
        E::A(val) | E::B(val) => match val {
            Some(n) => foo(n),
            _ => return,
        },
        _ => return,
    }
    match make::<Option<E<u32>>>() {
        Some(val) => match val {
            E::A(val) | E::B(val) => foo(val),
            _ => return,
        },
        _ => return,
    }
}

fn make<T>() -> T {
    unimplemented!()
}

fn foo<T, U>(t: T) -> U {
    unimplemented!()
}

fn take<T>(t: T) {}

fn main() {}
