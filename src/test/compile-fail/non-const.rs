// Test that various non const things are rejected.

fn foo<T: const>(_x: T) { }

struct r {
  let x:int;
  drop {}
}

fn r(x:int) -> r {
    r {
        x: x
    }
}

struct r2 {
  let x:@mut int;
  drop {}
}

fn r2(x:@mut int) -> r2 {
    r2 {
        x: x
    }
}

fn main() {
    foo({f: 3});
    foo({mut f: 3}); //~ ERROR missing `const`
    foo(~[1]);
    foo(~[mut 1]); //~ ERROR missing `const`
    foo(~1);
    foo(~mut 1); //~ ERROR missing `const`
    foo(@1);
    foo(@mut 1); //~ ERROR missing `const`
    foo(r(1)); // this is okay now.
    foo(r2(@mut 1)); //~ ERROR missing `const`
    foo({f: {mut f: 1}}); //~ ERROR missing `const`
}
