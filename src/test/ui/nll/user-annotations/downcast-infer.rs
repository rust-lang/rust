// build-pass (FIXME(62277): could be check-pass?)

// Check that we don't try to downcast `_` when type-checking the annotation.
fn main() {
    let x = Some(Some(Some(1)));

    match x {
        Some::<Option<_>>(Some(Some(v))) => (),
        _ => (),
    }
}
