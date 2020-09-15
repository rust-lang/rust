#[cfg(any())]
extern "C++" {
    #[namespace = std::experimental]
    type any;

    #[rust = std::option::Option<T>]
    //~^ ERROR expected one of `::` or `]`, found `<`
    type optional;
}

fn main() {}
