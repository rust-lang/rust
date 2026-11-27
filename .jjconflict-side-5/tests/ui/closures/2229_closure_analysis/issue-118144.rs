// Regression test for ICE #118144

struct V(i32);

fn func(func_arg: &mut V) {
    || {
        // Declaring `x` separately instead of using
        // a destructuring binding like `let V(x) = ...`
        // because only `V(x) = ...` triggers the ICE
        let x;
        V(x) = func_arg; //~ ERROR: mismatched types
        func_arg.0 = 0;
     };
}

fn main() {}
