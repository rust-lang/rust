// xfail-test

// Tests that shapes respect linearize_ty_params().

tag option<T> {
    none;
    some(T);
}

type smallintmap<T> = @{mutable v: [mutable option<T>]};

fn mk<@T>() -> smallintmap<T> {
    let v: [mutable option<T>] = [mutable];
    ret @{mutable v: v};
}

fn f<@T,@U>() {
    let sim = mk::<U>();
    log_err sim;
}

fn main() {
    f::<int,int>();
}

