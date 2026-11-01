struct S {
    a: u32,
}

fn main() {
    let s1 = S { a: 1 };

    let _ = || {
        let s2 = Oops { a: 2, ..s1 };
        //~^ ERROR cannot find struct, variant or union type `Oops` in this scope
    };
}
