fn foo<T>() where T: Default -> {
//~^ ERROR expected one of `(`, `+`, `,`, `::`, `<`, or `{`, found `->`
    0
}

fn main() {}
