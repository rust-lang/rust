struct SmolStr;

const _: fn() = || {
    match Some(()) {
        Some(()) => (),
        None => return,
    };
    let _: String = {
        SmolStr
        //~^ ERROR mismatched types
    };
};

fn main() {}
