struct NotCopyable;
#[derive(Clone)]
struct NotCopyableButCloneable;

fn func<F: FnMut() -> H, H: FnMut()>(_: F) {}

fn foo() {
    let mut var = None;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` here
        move || {
        //~^ ERROR: cannot move out of `var`
            var = Some(NotCopyable);
        }
    });
}
fn bar() {
    let mut var = None;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` nor to `clone()` here
        move || {
        //~^ ERROR: cannot move out of `var`
            var = Some(NotCopyableButCloneable);
        }
    });
}

fn main() {}
