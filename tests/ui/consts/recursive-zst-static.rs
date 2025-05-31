//@ revisions: default unleash
//@[unleash]compile-flags: -Zunleash-the-miri-inside-of-you

// This test ensures that we do not allow ZST statics to initialize themselves without ever
// actually creating a value of that type. This is important, as the ZST may have private fields
// that users can reasonably expect to only get initialized by their own code. Thus unsafe code
// can depend on this fact and will thus do unsound things when it is violated.
// See https://github.com/rust-lang/rust/issues/71078 for more details.

static FOO: () = FOO;
//~^ ERROR encountered static that tried to initialize itself with itself

static A: () = B; //~ ERROR cycle detected when evaluating initializer of static `A`
static B: () = A;

fn main() {
    FOO
}
