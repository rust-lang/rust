const fn foo() { }
const x: const fn() = foo;
//~^ ERROR `const fn` pointer type is unstable

static y: const fn() = foo;
//~^ ERROR `const fn` pointer type is unstable

const fn bar(f: const fn()) { f() }
//~^ ERROR `const fn` pointer type is unstable

struct Foo { field: const fn() }
//~^ ERROR `const fn` pointer type is unstable

fn main() {
  let local: fn() = foo;
  let local2: const fn() = foo;
  //~^ ERROR `const fn` pointer type is unstable
  let local3 = foo;
}
