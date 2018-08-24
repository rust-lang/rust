fn foo() -> u32 {
    return 'label: loop { break 'label 42; };
}

fn bar() -> u32 {
    loop { break 'label: loop { break 'label 42; }; }
    //~^ ERROR expected identifier, found keyword `loop`
    //~| ERROR expected type, found keyword `loop`
}

pub fn main() {
    foo();
}
