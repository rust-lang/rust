fn test_drop_replace() {
    let b: Box<isize>;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut b
    b = Box::new(1);    //~ NOTE first assignment
    b = Box::new(2);    //~ ERROR cannot assign twice to immutable variable `b`
                        //~| NOTE cannot assign twice to immutable
                        //~| NOTE in this expansion of desugaring of drop and replace
}

fn test_call() {
    let b = Box::new(1);    //~ NOTE first assignment
                            //~| HELP consider making this binding mutable
                            //~| SUGGESTION mut b
    b = Box::new(2);        //~ ERROR cannot assign twice to immutable variable `b`
                            //~| NOTE cannot assign twice to immutable
                            //~| NOTE in this expansion of desugaring of drop and replace
}

fn test_args(b: Box<i32>) {  //~ HELP consider making this binding mutable
                                //~| SUGGESTION mut b
    b = Box::new(2);            //~ ERROR cannot assign to immutable argument `b`
                                //~| NOTE cannot assign to immutable argument
                                //~| NOTE in this expansion of desugaring of drop and replace
}

fn main() {}
