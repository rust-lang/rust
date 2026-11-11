struct A;
struct B;
struct C;
struct D;

fn f(
    a1: A,
    a2: A,
    b1: B,
    b2: B,
    c1: C,
    c2: C,
) {}

fn main() {
    f(C, A, A, A, B, B, C); //~ ERROR function takes 6 arguments but 7 arguments were supplied [E0061]
    f(C, C, A, A, B, B);  //~ ERROR arguments to this function are incorrect [E0308]
    f(A, A, D, D, B, B);  //~ ERROR arguments to this function are incorrect [E0308]
    f(C, C, B, B, A, A);  //~ ERROR arguments to this function are incorrect [E0308]
    f(C, C, A, B, A, A);  //~ ERROR arguments to this function are incorrect [E0308]
}
