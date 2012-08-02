// Tests that shapes respect linearize_ty_params().

enum option<T> {
    none,
    some(T),
}

type smallintmap<T> = @{mut v: ~[mut option<T>]};

fn mk<T>() -> smallintmap<T> {
    let v: ~[mut option<T>] = ~[mut];
    return @{mut v: v};
}

fn f<T,U>() {
    let sim = mk::<U>();
    log(error, sim);
}

fn main() {
    f::<int,int>();
}

