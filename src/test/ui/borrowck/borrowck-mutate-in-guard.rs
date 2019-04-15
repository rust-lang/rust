enum Enum<'a> {
    A(&'a isize),
    B(bool),
}

fn foo() -> isize {
    let mut n = 42;
    let mut x = Enum::A(&mut n);
    match x {
        Enum::A(_) if { x = Enum::B(false); false } => 1,
        //~^ ERROR cannot assign in a pattern guard
        //~| WARN cannot assign `x` in match guard
        //~| WARN this error has been downgraded to a warning for backwards compatibility
        //~| WARN this represents potential undefined behavior in your code and this warning will
        Enum::A(_) if { let y = &mut x; *y = Enum::B(false); false } => 1,
        //~^ ERROR cannot mutably borrow in a pattern guard
        //~| ERROR cannot assign in a pattern guard
        //~| WARN cannot mutably borrow `x` in match guard
        //~| WARN this error has been downgraded to a warning for backwards compatibility
        //~| WARN this represents potential undefined behavior in your code and this warning will
        Enum::A(p) => *p,
        Enum::B(_) => 2,
    }
}

fn main() {
    foo();
}
