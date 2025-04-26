// Regression test for #69307
//
// Having a `async { .. foo.await .. }` block appear inside of a `+=`
// expression was causing an ICE due to a failure to save/restore
// state in the AST numbering pass when entering a nested body.
//
//@ check-pass
//@ edition:2018

fn block_on<F>(_: F) -> usize {
    0
}

fn main() {}

async fn bar() {
    let mut sum = 0;
    sum += {
        block_on(async {
            baz().await;
            let mut inner = 1;
            inner += block_on(async {
                baz().await;
                0
            })
        })
    };
}

async fn baz() {}
