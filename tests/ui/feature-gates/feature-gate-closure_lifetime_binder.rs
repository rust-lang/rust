//@ edition: 2024

fn main() {
    for<> || -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
    for<'a> || -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
    for<'a, 'b> |_: &'a ()| -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental

    // Issue #160431: suggest moving the binder onto a `fn` pointer binding type.
    let _cl = for<'a> |x: &'a str| -> (&'a str, &'a str) { x.split_at(0) };
    //~^ ERROR `for<...>` binders for closures are experimental

    // Local temporaries in the body are fine for `fn` pointers (machine-applicable).
    let _tmp = for<'a> |x: &'a str| -> usize {
        //~^ ERROR `for<...>` binders for closures are experimental
        let n = x.len();
        n
    };

    // Already has a type ascription — fall back to the simple help.
    let _typed: _ = for<'a> |x: &'a str| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental

    // Infer placeholders must not be copied into a MachineApplicable `fn` type.
    let _ret_infer = for<'a> |x: &'a str| -> _ { x };
    //~^ ERROR `for<...>` binders for closures are experimental
    //~| ERROR implicit types in closure signatures are forbidden when `for<...>` is present
    let _nested_infer = for<'a> |x: &'a _| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental
    //~| ERROR implicit types in closure signatures are forbidden when `for<...>` is present

    // Explicit `move` closures are not `fn` pointers.
    let y = 1;
    let _move = for<'a> move |x: &'a i32| -> i32 { *x + y };
    //~^ ERROR `for<...>` binders for closures are experimental

    // Possible captures (any case) still get a suggestion, but only as maybe-incorrect.
    let z = 1;
    let _capture = for<'a> |x: &'a i32| -> i32 { *x + z };
    //~^ ERROR `for<...>` binders for closures are experimental
    let Y = 1;
    let _upper = for<'a> |x: &'a i32| -> i32 { *x + Y };
    //~^ ERROR `for<...>` binders for closures are experimental

    // `if let` bindings must not escape into the `else` branch (or past the `if`).
    let if_let_env = 1;
    let _if_let = for<'a> |x: &'a i32| -> i32 {
        //~^ ERROR `for<...>` binders for closures are experimental
        if let Some(if_let_env) = None::<i32> {
            if_let_env
        } else {
            *x + if_let_env
        }
    };

    // Same for `while let`.
    let while_let_env = 1;
    let _while_let = for<'a> |x: &'a i32| -> i32 {
        //~^ ERROR `for<...>` binders for closures are experimental
        while let Some(while_let_env) = None::<i32> {
            let _ = while_let_env;
            break;
        }
        *x + while_let_env
    };

    // Let-chain bindings are scoped to the `if` as well (same name as the outer capture).
    let chain_env = 1;
    let _let_chain = for<'a> |x: &'a i32| -> i32 {
        //~^ ERROR `for<...>` binders for closures are experimental
        if let Some(chain_env) = None::<i32>
            && chain_env == 0
        {
            0
        } else {
            *x + chain_env
        }
    };

    // Locals declared in a `let else` block must not leak past it.
    let let_else_env = 1;
    let _let_else = for<'a> |x: &'a i32| -> i32 {
        //~^ ERROR `for<...>` binders for closures are experimental
        let Some(_) = None::<i32> else {
            let let_else_env = 0;
            return let_else_env;
        };
        *x + let_else_env
    };

    // Free functions look like captures to the AST heuristic; suggestion is maybe-incorrect.
    let _freefn = for<'a> |x: &'a i32| -> i32 { add(*x, 1) };
    //~^ ERROR `for<...>` binders for closures are experimental

    // `ref` bindings on the `let` are not rewritten.
    let ref _ref_cl = for<'a> |x: &'a str| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental

    // `ref` closure parameters are not rewritten.
    let _ref_param = for<'a> |ref x: &'a str| -> &'a str { *x };
    //~^ ERROR `for<...>` binders for closures are experimental

    // Parameter attributes would be dropped by the rewrite — fall back.
    let _attrs = for<'a> |#[allow(unused)] x: &'a str| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental

    // Non-lifetime binders are not valid on `fn` pointers.
    let _ty_binder = for<T> |x: T| -> T { x };
    //~^ ERROR `for<...>` binders for closures are experimental
    //~| ERROR only lifetime parameters can be used in this context

    // Bounded lifetime binders are not valid on `fn` pointers.
    let _bound = for<'a: 'static> |x: &'a str| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental
    //~| ERROR bounds cannot be used in this context

    // Pre-expansion gating still rejects binders under `#[cfg(false)]`.
    #[cfg(false)]
    let _cfg = for<'a> |x: &'a str| -> &'a str { x };
    //~^ ERROR `for<...>` binders for closures are experimental
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
