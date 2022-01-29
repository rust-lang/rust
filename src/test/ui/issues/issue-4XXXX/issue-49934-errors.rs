fn foo<#[derive(Debug)] T>() { //~ ERROR expected non-macro attribute, found attribute macro
    match 0 {
        #[derive(Debug)] //~ ERROR expected non-macro attribute, found attribute macro
        _ => (),
    }
}

fn main() {}
