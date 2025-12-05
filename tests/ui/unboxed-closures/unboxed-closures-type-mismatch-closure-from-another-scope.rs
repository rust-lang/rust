fn test() {
    let x = match **x { //~ ERROR
        Some(&a) if { panic!() } => {}
    };
    let mut p = &x;

    {
        let mut closure = expect_sig(|p, y| *p = y);
        closure(&mut p, &y); //~ ERROR
        //~^ ERROR
    }

    deref(p); //~ ERROR
}

fn expect_sig<F>(f: F) -> F
where
    F: FnMut(&mut &i32, &i32),
{
    f
}

fn main() {}
