#[derive(Debug)]
struct NotCopyable;
#[derive(Debug, Clone)]
struct NotCopyableButCloneable;

fn func<F: FnMut() -> H, H: FnMut()>(_: F) {}

fn foo() {
    let var = NotCopyable;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` here
        move || { //~ ERROR cannot move out of `var`
            let x = var; //~ ERROR cannot move out of `var`
            println!("{x:?}");
        }
    });
}

fn bar() {
    let var = NotCopyableButCloneable;
    func(|| {
        // Shouldn't suggest `move ||.as_ref()` here
        move || { //~ ERROR cannot move out of `var`
            let x = var; //~ ERROR cannot move out of `var`
            println!("{x:?}");
        }
    });
}

fn main() {}
