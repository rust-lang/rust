// Regression test for #70934

struct S;

fn foo() {
    &([S][0],);
    //~^ ERROR cannot move out of type `[S; 1]`
}

fn main() {}
