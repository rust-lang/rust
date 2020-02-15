fn main() {}

extern {
    const A: isize;
    //~^ ERROR extern items cannot be `const`
    const B: isize = 42;
    //~^ ERROR extern items cannot be `const`
}
