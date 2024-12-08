// test for #117058 - check that attributes are validated on various kinds of statements.

struct A;

fn func() {}

fn main() {
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    if true {
    } else {
    }
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    (1);
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    match 1 {
        _ => {}
    }
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    while false {}
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    {}
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    A {};
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    func();
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    A;
    #[allow(two-words)]
    //~^ ERROR expected one of `(`, `,`, `::`, or `=`, found `-`
    loop {}
}
