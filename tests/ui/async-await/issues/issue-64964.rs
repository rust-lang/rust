// check-pass
// incremental
// compile-flags: -Z query-dep-graph
// edition:2018

// Regression test for ICE related to `await`ing in a method + incr. comp. (#64964)

struct Body;
impl Body {
    async fn next(&mut self) {
        async {}.await
    }
}

// Another reproduction: `await`ing with a variable from for-loop.

async fn bar() {
    for x in 0..10 {
        async { Some(x) }.await.unwrap();
    }
}

fn main() {}
