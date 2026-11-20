//@ revisions: label note both
//@[both] compile-flags: --cfg label --cfg note
//@ dont-require-annotations: NOTE

//! Example and test of multiple diagnostic attributes coalescing together.

#[diagnostic::on_unimplemented(message = "this message", note = "a note")]
#[cfg_attr(label, diagnostic::on_unimplemented(label = "this label"))]
#[cfg_attr(note, diagnostic::on_unimplemented(note = "another note"))]
trait Trait {}

fn takes_trait(_: impl Trait) {}

fn main() {
    takes_trait(());
    //~^ERROR this message
    //[label]~|NOTE this label
    //[both]~|NOTE this label
    //~|NOTE a note
    //[note]~|NOTE another note
    //[both]~|NOTE another note
}
