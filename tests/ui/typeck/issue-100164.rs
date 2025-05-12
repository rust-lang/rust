//@ run-rustfix

const _A: = 123;
//~^ ERROR: missing type for `const` item

fn main() {
    const _B: = 123;
    //~^ ERROR: missing type for `const` item
}
