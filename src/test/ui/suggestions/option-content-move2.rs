struct NotCopyable;

fn func<F: FnMut() -> H, H: FnMut()>(_: F) {}

fn parse() {
    let mut var = None;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` here
        move || {
        //~^ ERROR: cannot move out of `var`
            var = Some(NotCopyable);
        }
    });
}

fn main() {}
