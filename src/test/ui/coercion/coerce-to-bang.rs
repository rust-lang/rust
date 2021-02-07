#![feature(never_type)]

fn foo(x: usize, y: !, z: usize) { }

fn call_foo_a() {
    foo(return, 22, 44);
    //~^ ERROR arguments to this function are incorrect
}

fn call_foo_b() {
    // Divergence happens in the argument itself, definitely ok.
    foo(22, return, 44);
}

fn call_foo_c() {
    // This test fails because the divergence happens **after** the
    // coercion to `!`:
    foo(22, 44, return); //~ ERROR arguments to this function are incorrect
}

fn call_foo_d() {
    // This test passes because `a` has type `!`:
    let a: ! = return;
    let b = 22;
    let c = 44;
    foo(a, b, c); // ... and hence a reference to `a` is expected to diverge.
    //~^ ERROR arguments to this function are incorrect
}

fn call_foo_e() {
    // This test probably could pass but we don't *know* that `a`
    // has type `!` so we don't let it work.
    let a = return;
    let b = 22;
    let c = 44;
    foo(a, b, c); //~ ERROR arguments to this function are incorrect
}

fn call_foo_f() {
    // This fn fails because `a` has type `usize`, and hence a
    // reference to is it **not** considered to diverge.
    let a: usize = return;
    let b = 22;
    let c = 44;
    foo(a, b, c); //~ ERROR arguments to this function are incorrect
}

fn array_a() {
    // Return is coerced to `!` just fine, but `22` cannot be.
    let x: [!; 2] = [return, 22]; //~ ERROR arguments to this function are incorrect
}

fn array_b() {
    // Error: divergence has not yet occurred.
    let x: [!; 2] = [22, return]; //~ ERROR arguments to this function are incorrect
}

fn tuple_a() {
    // No divergence at all.
    let x: (usize, !, usize) = (22, 44, 66);
    //~^ ERROR arguments to this function are incorrect
}

fn tuple_b() {
    // Divergence happens before coercion: OK
    let x: (usize, !, usize) = (return, 44, 66);
    //~^ ERROR arguments to this function are incorrect
}

fn tuple_c() {
    // Divergence happens before coercion: OK
    let x: (usize, !, usize) = (22, return, 66);
}

fn tuple_d() {
    // Error: divergence happens too late
    let x: (usize, !, usize) = (22, 44, return);
    //~^ ERROR arguments to this function are incorrect
}

fn main() { }
