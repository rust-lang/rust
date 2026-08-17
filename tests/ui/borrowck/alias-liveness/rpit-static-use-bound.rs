//@ check-pass

fn foo<'a>(x: &'a mut i32) -> impl Sized + use<'a> + 'static {}

fn overlapping_mut() {
    let i = &mut 1;
    let x = foo(i);
    let y = foo(i);
}

fn live_past_borrow() {
    let y;
    {
        let x = &mut 1;
        y = foo(x);
    }
}

fn main() {}
