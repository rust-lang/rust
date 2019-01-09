// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test_drop_replace() {
    let b: Box<isize>;
    //[mir]~^ HELP make this binding mutable
    //[mir]~| SUGGESTION mut b
    b = Box::new(1);    //[ast]~ NOTE first assignment
                        //[mir]~^ NOTE first assignment
    b = Box::new(2);    //[ast]~ ERROR cannot assign twice to immutable variable
                        //[mir]~^ ERROR cannot assign twice to immutable variable `b`
                        //[ast]~| NOTE cannot assign twice to immutable
                        //[mir]~| NOTE cannot assign twice to immutable
}

fn test_call() {
    let b = Box::new(1);    //[ast]~ NOTE first assignment
                            //[mir]~^ NOTE first assignment
                            //[mir]~| HELP make this binding mutable
                            //[mir]~| SUGGESTION mut b
    b = Box::new(2);        //[ast]~ ERROR cannot assign twice to immutable variable
                            //[mir]~^ ERROR cannot assign twice to immutable variable `b`
                            //[ast]~| NOTE cannot assign twice to immutable
                            //[mir]~| NOTE cannot assign twice to immutable
}

fn test_args(b: Box<i32>) {  //[ast]~ NOTE first assignment
                                //[mir]~^ HELP make this binding mutable
                                //[mir]~| SUGGESTION mut b
    b = Box::new(2);            //[ast]~ ERROR cannot assign twice to immutable variable
                                //[mir]~^ ERROR cannot assign to immutable argument `b`
                                //[ast]~| NOTE cannot assign twice to immutable
                                //[mir]~| NOTE cannot assign to immutable argument
}

fn main() {}
