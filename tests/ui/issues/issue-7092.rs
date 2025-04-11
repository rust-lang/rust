enum Whatever {
}

fn foo(x: Whatever) {
    match x {
        Some(field) =>
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected `Whatever`, found `Option<_>`
//~| NOTE_NONVIRAL expected enum `Whatever`
//~| NOTE_NONVIRAL found enum `Option<_>`
            field.access(),
    }
}

fn main(){}
