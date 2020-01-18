#![feature(never_type)]
#![feature(never_type_fallback)]

fn get_type<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

fn unconstrained_return<T>() -> Result<T, String> {
    Err("Hi".to_string())
}

fn foo() {
    let a = || {
        match unconstrained_return::<_>() { //~ ERROR Fallback to `!` may introduce undefined
            Ok(x) => x,  // `x` has type `_`, which is unconstrained
            Err(s) => panic!(s),  // … except for unifying with the type of `panic!()`
            // so that both `match` arms have the same type.
            // Therefore `_` resolves to `!` and we "return" an `Ok(!)` value.
        }
    };

    let cast: &dyn FnOnce() -> _ = &a;
    println!("Return type: {:?}", get_type(cast));
}

fn main() {
    foo()
}
