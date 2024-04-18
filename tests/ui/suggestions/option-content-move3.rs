#[derive(Debug, Clone)]
struct NotCopyable;

fn func<F: FnMut() -> H, H: FnMut()>(_: F) {}

fn parse() {
    let mut var = NotCopyable;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` here
        move || { //~ ERROR cannot move out of `var`
            let x = var; //~ ERROR cannot move out of `var`
            println!("{x:?}");
        }
    });
}

fn main() {}
