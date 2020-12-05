fn foo<#[derive(Debug)] T>() {
//~^ ERROR `derive` may only be applied to structs, enums and unions
    match 0 {
        #[derive(Debug)]
        //~^ ERROR `derive` may only be applied to structs, enums and unions
        _ => (),
    }
}

fn main() {
}
