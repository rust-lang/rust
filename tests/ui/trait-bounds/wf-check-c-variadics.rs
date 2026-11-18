// In `hir_wf_check.rs` we try to find the innermost type that is not well formed,
// so we can give the user an accurate span.
// This test checks if this happens correctly for c-variadics.

struct NeedsCopy<T: Copy>(T);

unsafe extern "C" {
    fn f(x: i32, ...) -> Option<NeedsCopy<String>>;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn main() {}
