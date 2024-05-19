//@ edition:2018

extern {
    async fn L() { //~ ERROR: incorrect function inside `extern` block
        //~^ ERROR: functions in `extern` blocks cannot have qualifiers
        async fn M() {}
    }
}

fn main() {}
