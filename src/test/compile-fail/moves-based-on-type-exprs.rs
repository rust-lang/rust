// Tests that references to move-by-default values trigger moves when
// they occur as part of various kinds of expressions.

struct Foo<A> { f: A }
fn guard(_s: ~str) -> bool {fail2!()}
fn touch<A>(_a: &A) {}

fn f10() {
    let x = ~"hi";
    let _y = Foo { f:x };
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f20() {
    let x = ~"hi";
    let _y = (x, 3);
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f21() {
    let x = ~[1, 2, 3];
    let _y = (x[0], 3);
    touch(&x);
}

fn f30(cond: bool) {
    let x = ~"hi";
    let y = ~"ho";
    let _y = if cond {
        x
    } else {
        y
    };
    touch(&x); //~ ERROR use of moved value: `x`
    touch(&y); //~ ERROR use of moved value: `y`
}

fn f40(cond: bool) {
    let x = ~"hi";
    let y = ~"ho";
    let _y = match cond {
        true => x,
        false => y
    };
    touch(&x); //~ ERROR use of moved value: `x`
    touch(&y); //~ ERROR use of moved value: `y`
}

fn f50(cond: bool) {
    let x = ~"hi";
    let y = ~"ho";
    let _y = match cond {
        _ if guard(x) => 10,
        true => 10,
        false => 20,
    };
    touch(&x); //~ ERROR use of moved value: `x`
    touch(&y);
}

fn f70() {
    let x = ~"hi";
    let _y = [x];
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f80() {
    let x = ~"hi";
    let _y = ~[x];
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f90() {
    let x = ~"hi";
    let _y = @[x];
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f100() {
    let x = ~[~"hi"];
    let _y = x[0];
    touch(&x); //~ ERROR use of partially moved value: `x`
}

fn f110() {
    let x = ~[~"hi"];
    let _y = [x[0], ..1];
    touch(&x); //~ ERROR use of partially moved value: `x`
}

fn f120() {
    let mut x = ~[~"hi", ~"ho"];
    x.swap(0, 1);
    touch(&x[0]);
    touch(&x[1]);
}

fn main() {}
