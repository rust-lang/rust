//@ edition:2024

fn f0(){
    r#"ok0!"#;
}

fn f1(){
    r#"ok1!"##;
    //~^ ERROR too many `#` when terminating raw string
}


fn f2(){
    r#"ok2!"###;
    //~^ ERROR reserved multi-hash token is forbidden
    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `##`
}

fn f3(){
    #"ok3!"#;
    //~^ ERROR invalid string literal
}


fn f4(){
    #"ok4!"##;
    //~^ ERROR invalid string literal
    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `#`
}

fn f5(){
    #"ok5!"###;
    //~^ ERROR invalid string literal
    //~| ERROR reserved multi-hash token is forbidden
    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `##`
}

fn main() {}
