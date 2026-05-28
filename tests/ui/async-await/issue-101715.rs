//@ edition:2018

struct S;

impl S {
    fn very_long_method_name_the_longest_method_name_in_the_whole_universe(self) {}
}

async fn foo() {
    S.very_long_method_name_the_longest_method_name_in_the_whole_universe()
        .await
        //~^ error: `()` is not a future
        //~| help: remove the `.await`
        //~| help: the trait `Future` is not implemented for `()`
}

fn main() {}
