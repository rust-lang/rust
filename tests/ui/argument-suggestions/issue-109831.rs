struct A;
struct B;

fn f(b1: B, b2: B, a2: C) {} //~ ERROR E0412

fn main() {
    f(A, A, B, C); //~ ERROR E0425
    //~^ ERROR E0061
}
