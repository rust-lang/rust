#[allow(unused_macros)]
macro_rules! foo {
    () => {

    }
    () => {
        //~^ ERROR expected `;`, found `(`
    }
}

fn main() {}
