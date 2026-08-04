//@ edition: 2024
//@ compile-flags: --error-format=json
//@ error-pattern: "suggestion_applicability":"MaybeIncorrect"

// Capturing closures must not get a MachineApplicable rewrite. Cover plain captures, let-else
// leakage, and let-chain shadowing — all should report MaybeIncorrect in JSON.

fn main() {
    let y = 1;
    let _capture = for<'a> |x: &'a i32| -> i32 { *x + y };
    //~^ ERROR `for<...>` binders for closures are experimental

    let let_else_env = 1;
    let _let_else = for<'a> |x: &'a i32| -> i32 {
        //~^ ERROR `for<...>` binders for closures are experimental
        let Some(_) = None::<i32> else {
            let let_else_env = 0;
            return let_else_env;
        };
        *x + let_else_env
    };

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
}
