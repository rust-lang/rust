// Make sure const bounds work on things, and test that a few types
// are const.


fn foo<T: copy const>(x: T) -> T { x }

fn main() {
    foo(1);
    foo("hi");
    foo([1, 2, 3]/~);
    foo({field: 42});
    foo((1, 2u));
    foo(@1);
    foo(~1);
}
