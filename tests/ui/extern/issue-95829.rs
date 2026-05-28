//@ edition:2018
//@ reference: items.extern.fn.qualifiers

extern "C" {
    async fn L() { //~ ERROR: incorrect function inside `extern` block
        //~^ ERROR: functions in `extern` blocks cannot have `async` qualifier
        async fn M() {}
    }
}

fn main() {}
