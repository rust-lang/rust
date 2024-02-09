#![feature(inline_const_pat)]

const unsafe fn require_unsafe() -> usize {
    1
}

fn main() {
    match () {
        const {
            require_unsafe();
            //~^ ERROR [E0133]
        } => (),
    }

    match 1 {
        const {
            require_unsafe()
            //~^ ERROR [E0133]
        }..=4 => (),
        _ => (),
    }
}
