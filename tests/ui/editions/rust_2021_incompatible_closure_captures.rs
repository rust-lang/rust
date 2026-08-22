//@ edition:2018

#[warn(rust_2021_incompatible_closure_captures)]
enum Functions {
    Square = |b| move || format!("{b}"),
    //~^ ERROR mismatched types
    //~| WARN changes to closure capture in Rust 2021 will affect drop order
}

#[warn(rust_2021_incompatible_closure_captures)]
static _static: () = |b| move || b;
//~^ ERROR mismatched types
//~| WARN changes to closure capture in Rust 2021 will affect drop order

fn main() {
    #[warn(rust_2021_incompatible_closure_captures)]
    const _: () = |b| move || b;
    //~^ ERROR mismatched types
    //~| WARN changes to closure capture in Rust 2021 will affect drop order

    #[warn(rust_2021_incompatible_closure_captures)]
    let _: () = |b| move || b;
    //~^ ERROR mismatched types
    //~| WARN changes to closure capture in Rust 2021 will affect drop order
}
