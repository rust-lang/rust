// Check the "dropped here, when the enclosing ... returns" label for different body kinds.
//@ edition: 2024

fn foo(_: &impl Fn(&u32) -> &u32) {}

async fn in_async_fn() {
    let x = 22;
    foo(&|_| &x)
    //~^ ERROR `x` does not live long enough
}

fn in_async_block() {
    let _ = async {
        let x = 22;
        foo(&|_| &x)
        //~^ ERROR `x` does not live long enough
    };
}

fn in_closure() {
    let _ = || {
        let x = 22;
        foo(&|_| &x)
        //~^ ERROR `x` does not live long enough
    };
}

fn main() {}
