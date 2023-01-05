struct A;

impl Drop for A {
    fn drop(&mut self) {}
}

fn takes_a_ref<'a>(_arg: &'a A) {}

fn returns_a() -> A {
    A
}

fn weird_temporary<'a>(a: &'a A, x: bool) {
    takes_a_ref('scope: {
        if x {
            break 'scope a;
        }

        &returns_a()
        //~^ ERROR temporary value dropped while borrowed
    });
}

fn main() {}
