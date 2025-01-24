macro_rules! arm {
    ($pattern:pat => $block:block) => {
        $pattern => $block
        //~^ ERROR macro expansion ignores `=>` and any tokens following
        //~| NOTE the usage of `arm!` is likely invalid in pattern context
        //~| NOTE macros cannot expand to match arms
    };
}

fn main() {
    let x = Some(1);
    match x {
        Some(1) => {},
        arm!(None => {}),
        //~^ NOTE caused by the macro expansion here
        //~| ERROR `match` arm with no body
        Some(2) => {},
        _ => {},
    };
}
