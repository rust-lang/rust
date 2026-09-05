//@ check-pass

trait Trait<'a>: 'a {}
impl Trait<'_> for () {}

fn scope0<'a>() {
    // NOTE: Used to be elab'ed to `dyn Trait<'a> + 'a`, now it's `dyn Trait<'a> + '_`.
    let bx: Box<dyn Trait<'a>> = Box::new(());
    let _: Box<dyn Trait<'a> + 'static> = bx;
}

fn scope1<'a>() {
    // NOTE: Used to be elab'ed to `dyn Trait<'a> + 'a`, now it's `dyn Trait<'a> + '_`.
    let bx: Box<dyn Trait<'a> + '_> = Box::new(());
    let _: Box<dyn Trait<'a> + 'static> = bx;
}

fn main() {}
