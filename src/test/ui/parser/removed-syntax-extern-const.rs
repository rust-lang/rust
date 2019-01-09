extern {
    const i: isize;
    //~^ ERROR extern items cannot be `const`
}

fn main() {}
