// Regression test for #123154

struct AA {
    pub data: [&usize]
    //~^ ERROR missing lifetime specifier
}

impl AA {
    const fn new() -> Self { }
    //~^ ERROR mismatched types
}

static ST: AA = AA::new();

fn main() {}
