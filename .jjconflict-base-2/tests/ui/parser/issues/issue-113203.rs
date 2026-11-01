// Checks what happens when we attempt to use the await keyword as a prefix. Span
// incorrectly emitted an `.await` in E0277 which does not exist
//@ edition:2018
fn main() {
    await {}()
    //~^ ERROR incorrect use of `await`
}
