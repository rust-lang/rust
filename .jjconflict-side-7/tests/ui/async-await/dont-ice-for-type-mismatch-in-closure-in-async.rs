//@ edition: 2021

fn call(_: impl Fn() -> bool) {}

async fn test() {
    call(|| -> Option<()> {
        //~^ ERROR expected
        if true {
            false
            //~^ ERROR mismatched types
        }
        true
        //~^ ERROR mismatched types
    })
}

fn main() {}
