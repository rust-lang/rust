struct A;
struct B;
struct C;
struct D;
struct E;
struct F;
struct G;

fn foo(a: &A, d: D, e: &E, g: G) {}

fn main() {
    foo(&&A, B, C, D, E, F, G);
    //~^ ERROR function takes 4 arguments but 7 arguments were supplied
}
