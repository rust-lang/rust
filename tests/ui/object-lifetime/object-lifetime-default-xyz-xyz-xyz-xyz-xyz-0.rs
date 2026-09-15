//@ check-pass

trait Trait {}
impl Trait for () {}

fn scope<'a>(arg: &'a ()) {
    // NOTE: Use to be elab'ed to `&'a (dyn Trait + 'a)`.
    //       Now elabs to `&'a (dyn Trait + '_)`.
    let dt: &'a dyn Trait = arg;
    let _: &(dyn Trait + 'static) = dt;
}

fn main() {}
