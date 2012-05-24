// Test that various non const things are rejected.

fn foo<T: const>(_x: T) { }

resource r(_x: int) {}

fn main() {
    foo({f: 3});
    foo({mut f: 3}); //! ERROR missing `const`
    foo([1]);
    foo([mut 1]); //! ERROR missing `const`
    foo(~1);
    foo(~mut 1); //! ERROR missing `const`
    foo(@1);
    foo(@mut 1); //! ERROR missing `const`
    foo(r(1)); //! ERROR missing `const`
    foo("123");
    foo({f: {mut f: 1}}); //! ERROR missing `const`
}
