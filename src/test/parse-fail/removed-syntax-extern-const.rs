// compile-flags: -Z parse-only

extern {
    const i: isize;
    //~^ ERROR extern items cannot be `const`
}
