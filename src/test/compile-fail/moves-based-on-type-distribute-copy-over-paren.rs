// Tests that references to move-by-default values trigger moves when
// they occur as part of various kinds of expressions.

struct Foo<A> { f: A }
fn touch<A>(_a: &A) {}

fn f00() {
    let x = ~"hi";
    let _y = Foo { f:x }; //~ NOTE `x` moved here
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f05() {
    let x = ~"hi";
    let _y = Foo { f:(((x))) }; //~ NOTE `x` moved here
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f10() {
    let x = ~"hi";
    let _y = Foo { f:x.clone() };
    touch(&x);
}

fn f20() {
    let x = ~"hi";
    let _y = Foo { f:(x).clone() };
    touch(&x);
}

fn f30() {
    let x = ~"hi";
    let _y = Foo { f:((x)).clone() };
    touch(&x);
}

fn f40() {
    let x = ~"hi";
    let _y = Foo { f:(((((((x)).clone()))))) };
    touch(&x);
}

fn main() {}
