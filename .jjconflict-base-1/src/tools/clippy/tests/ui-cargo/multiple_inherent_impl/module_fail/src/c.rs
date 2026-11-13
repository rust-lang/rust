pub struct S;
struct T;

impl S {
    fn first() {}
}

impl T {
    fn first() {}
}

impl T {
    //^ Must trigger
    fn second() {}
}
