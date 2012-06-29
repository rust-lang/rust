// Test that various non const things are rejected.

fn foo<T: const>(_x: T) { }

class r {
  let x:int;
  new(x:int) { self.x = x; }
  drop {}
}

class r2 {
  let x:@mut int;
  new(x:@mut int) { self.x = x; }
  drop {}
}

fn main() {
    foo({f: 3});
    foo({mut f: 3}); //! ERROR missing `const`
    foo(~[1]);
    foo(~[mut 1]); //! ERROR missing `const`
    foo(~1);
    foo(~mut 1); //! ERROR missing `const`
    foo(@1);
    foo(@mut 1); //! ERROR missing `const`
    foo(r(1)); // this is okay now.
    foo(r2(@mut 1)); //! ERROR missing `const`
    foo("123");
    foo({f: {mut f: 1}}); //! ERROR missing `const`
}
