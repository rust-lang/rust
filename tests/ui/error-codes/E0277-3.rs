fn foo<T: PartialEq>(_: T) {}

struct S;

fn main() {
    foo(S);
    //~^ ERROR can't compare `S` with `S`
}
