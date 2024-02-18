//@ edition:2018
//
// Regression test for issue #73741
// Ensures that we don't emit spurious errors when
// a type error ocurrs in an `async fn`

async fn weird() {
    1 = 2; //~ ERROR invalid left-hand side

    let mut loop_count = 0;
    async {}.await
}

fn main() {}
