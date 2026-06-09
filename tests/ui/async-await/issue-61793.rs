// This testcase used to ICE in codegen due to inconsistent field reordering
// in the coroutine state, claiming a ZST field was after a non-ZST field,
// while those two fields were at the same offset (which is impossible).
// That is, memory ordering of `(X, ())`, but offsets of `((), X)`.

//@ build-pass
//@ edition:2018

async fn foo<F>(_: &(), _: F) {}

fn main() {
    foo(&(), || {});
    async {
        foo(&(), || {}).await;
    };
}
