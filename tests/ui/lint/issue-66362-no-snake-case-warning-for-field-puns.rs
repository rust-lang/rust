#![deny(non_snake_case)]
#![allow(unused_variables)]
#![allow(dead_code)]

enum Foo {
    Bad {
        lowerCamelCaseName: bool,
        //~^ ERROR structure field `lowerCamelCaseName` should have a snake case name
    },
    Good {
        snake_case_name: bool,
    },
}

fn main() {
    let b = Foo::Bad { lowerCamelCaseName: true };

    match b {
        Foo::Bad { lowerCamelCaseName } => {}
        Foo::Good { snake_case_name: lowerCamelCaseBinding } => { }
        //~^ ERROR variable `lowerCamelCaseBinding` should have a snake case name
    }

    if let Foo::Good { snake_case_name: anotherLowerCamelCaseBinding } = b { }
    //~^ ERROR variable `anotherLowerCamelCaseBinding` should have a snake case name

    if let Foo::Bad { lowerCamelCaseName: yetAnotherLowerCamelCaseBinding } = b { }
    //~^ ERROR variable `yetAnotherLowerCamelCaseBinding` should have a snake case name
}
