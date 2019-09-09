fn foo<#[derive(Debug)] T>() {
//~^ ERROR `derive` may only be applied to structs, enums and unions
//~| ERROR expected an inert attribute, found an attribute macro
    match 0 {
        #[derive(Debug)]
        //~^ ERROR `derive` may only be applied to structs, enums and unions
        //~| ERROR expected an inert attribute, found an attribute macro
        _ => (),
    }
}

fn main() {
}
