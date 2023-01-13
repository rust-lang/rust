// Tests that references to move-by-default values trigger moves when
// they occur as part of various kinds of expressions.

struct Foo<A> { f: A }
fn touch<A>(_a: &A) {}

fn f00() {
    let x = "hi".to_string();
    //~^ NOTE move occurs because `x` has type `String`
    let _y = Foo { f:x };
    //~^ NOTE value moved here
    touch(&x); //~ ERROR borrow of moved value: `x`
    //~^ NOTE value borrowed here after move
}

fn f05() {
    let x = "hi".to_string();
    //~^ NOTE move occurs because `x` has type `String`
    let _y = Foo { f:(((x))) };
    //~^ NOTE value moved here
    touch(&x); //~ ERROR borrow of moved value: `x`
    //~^ NOTE value borrowed here after move
}

fn f10() {
    let x = "hi".to_string();
    let _y = Foo { f:x.clone() };
    touch(&x);
}

fn f20() {
    let x = "hi".to_string();
    let _y = Foo { f:(x).clone() };
    touch(&x);
}

fn f30() {
    let x = "hi".to_string();
    let _y = Foo { f:((x)).clone() };
    touch(&x);
}

fn f40() {
    let x = "hi".to_string();
    let _y = Foo { f:(((((((x)).clone()))))) };
    touch(&x);
}

fn main() {}
