//@ check-pass

trait Double<'a, 'b>: 'a + 'b {}
impl Double<'_, '_> for () {}

fn scope<'a, 'b>() {
    // NOTE: These used to get rejected due to "ambiguity".
    let _: Box<dyn Double<'a, 'b>> = Box::new(());
    let _: Box<dyn Double<'a, 'b> + '_> = Box::new(());
}

fn main() {}
