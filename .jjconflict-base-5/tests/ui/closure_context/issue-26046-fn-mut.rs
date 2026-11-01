fn foo() -> Box<dyn Fn()> {
    let num = 5;

    let closure = || { //~ ERROR expected a closure that
        num += 1;
    };

    Box::new(closure)
}

fn main() {}
