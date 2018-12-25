enum Whatever {
}

fn foo(x: Whatever) {
    match x {
        Some(field) =>
//~^ ERROR mismatched types
//~| expected type `Whatever`
//~| found type `std::option::Option<_>`
//~| expected enum `Whatever`, found enum `std::option::Option`
            field.access(),
    }
}

fn main(){}
