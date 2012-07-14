fn main() {

    // Note: here we do not have any type annotations
    // but we do express conflicting requirements:

    let v = ~[mut ~[0]];
    let w = ~[mut ~[mut 0]];
    let x = ~[mut ~[mut 0]];

    fn f(&&v: ~[mut ~[int]]) {
        v[0] = ~[3]
    }

    fn g(&&v: ~[const ~[const int]]) {
    }

    fn h(&&v: ~[mut ~[mut int]]) {
        v[0] = ~[mut 3]
    }

    fn i(&&v: ~[mut ~[const int]]) {
        v[0] = ~[mut 3]
    }

    fn j(&&v: ~[~[const int]]) {
    }

    f(v);
    g(v);
    h(v); //~ ERROR (values differ in mutability)
    i(v); //~ ERROR (values differ in mutability)
    j(v); //~ ERROR (values differ in mutability)

    f(w); //~ ERROR (values differ in mutability)
    g(w);
    h(w);
    i(w); //~ ERROR (values differ in mutability)
    j(w); //~ ERROR (values differ in mutability)

    // Note that without adding f() or h() to the mix, it is valid for
    // x to have the type ~[mut ~[const int]], and thus we can safely
    // call g() and i() but not j():
    g(x);
    i(x);
    j(x); //~ ERROR (values differ in mutability)
}
