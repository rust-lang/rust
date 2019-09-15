fn foo<#[derive(Debug)] T>() {
//~^ ERROR `derive` may only be applied to structs, enums and unions
//~| ERROR expected an inert attribute, found a derive macro
    match 0 {
        #[derive(Debug)]
        //~^ ERROR `derive` may only be applied to structs, enums and unions
        //~| ERROR expected an inert attribute, found a derive macro
        _ => (),
    }
}

fn main() {
}
