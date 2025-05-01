enum Whatever {
}

fn foo(x: Whatever) {
    match x { //~ NOTE this expression has type `Whatever`
        Some(field) =>
//~^ ERROR mismatched types
//~| NOTE expected `Whatever`, found `Option<_>`
//~| NOTE expected enum `Whatever`
//~| NOTE found enum `Option<_>`
            field.access(),
    }
}

fn main(){}
