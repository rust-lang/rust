enum Whatever {
}

fn foo(x: Whatever) {
    match x {
        Some(field) =>
//~^ ERROR mismatched types
//~| expected `Whatever`, found `Option<_>`
//~| expected enum `Whatever`
//~| found enum `Option<_>`
            field.access(),
    }
}

fn main(){}
