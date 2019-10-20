extern {
    const fn foo();
    //~^ ERROR extern items cannot be `const`
    const unsafe fn bar();
    //~^ ERROR extern items cannot be `const`
}

fn main() {}
