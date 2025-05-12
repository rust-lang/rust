trait Trait {
//~^ NOTE in this trait...
//~| NOTE in this trait...
    fn bar<'a,'b:'a>(x: &'a str, y: &'b str);
    //~^ NOTE `'a` is early-bound
    //~| NOTE this lifetime bound makes `'a` early-bound
    //~| NOTE `'b` is early-bound
    //~| NOTE this lifetime bound makes `'b` early-bound
}

struct Foo;

impl Trait for Foo {
//~^ NOTE in this impl...
//~| NOTE in this impl...
    fn bar<'a,'b>(x: &'a str, y: &'b str) {
    //~^ ERROR E0195
    //~| NOTE `'a` differs between the trait and impl
    //~| NOTE `'a` is late-bound
    //~| NOTE `'b` differs between the trait and impl
    //~| NOTE `'b` is late-bound
    //~| NOTE lifetime parameters differ in whether they are early- or late-bound
    }
}

fn main() {
}
