//@ edition:2021
struct AnyOption<T>(T);
impl<T> AnyOption<T> {
    const NONE: Option<T> = None;
}

fn uwu() {}
fn defines() {
    match Some(uwu) {
        AnyOption::<_>::NONE => {}
        //~^ ERROR constant of non-structural type
        _ => {}
    }

    match Some(|| {}) {
        AnyOption::<_>::NONE => {}
        //~^ ERROR constant of non-structural type
        _ => {}
    }
}
fn main() {}
