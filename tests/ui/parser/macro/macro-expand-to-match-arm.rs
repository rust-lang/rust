macro_rules! arm {
    ($pattern:pat => $block:block) => {
        $pattern => $block
    };
}

fn main() {
    let x = Some(1);
    match x {
        Some(1) => {},
        arm!(None => {}),
        //~^ NOTE macros cannot expand to match arms
        //~| ERROR unexpected `,` in pattern
        // doesn't recover
        Some(2) => {},
        _ => {},
    };
}
