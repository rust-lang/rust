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

const _: fn() -> SmolStr = || {
    match Some(()) {
        Some(()) => (),
        None => return SmolStr,
    };
    let _: String = {
        SmolStr
        //~^ ERROR mismatched types
    };
    SmolStr
};

const _: fn() -> String = || {
    match Some(()) {
        Some(()) => (),
        None => return String::new(),
    };
    let _: String = {
        SmolStr
        //~^ ERROR mismatched types
    };
    String::new()
};

fn main() {}
