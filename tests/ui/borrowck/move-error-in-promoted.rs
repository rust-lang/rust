// Regression test for #70934

fn f() {
    const C: [S2; 1] = [S2];
    let _ = S1(C[0]).clone();
    //~^ ERROR cannot move out of type `[S2; 1]`
}

#[derive(Clone)]
struct S1(S2);

#[derive(Clone)]
struct S2;

fn main() {
    f();
}
