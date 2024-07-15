use std::fs;
use std::path::Path;

/// Archive utility.
///
/// # Notes
///
/// This *currently* uses the [ar][rust-ar] crate, but this is subject to changes. We may need to
/// use `llvm-ar`, and if that is the case, this should be moved under `external_deps`.
///
/// [rust-ar]: https://github.com/mdsteele/rust-ar
#[track_caller]
pub fn ar(inputs: &[impl AsRef<Path>], output_path: impl AsRef<Path>) {
    let output = fs::File::create(&output_path).expect(&format!(
        "the file in path `{}` could not be created",
        output_path.as_ref().display()
    ));
    let mut builder = ar::Builder::new(output);
    for input in inputs {
        builder.append_path(input).unwrap();
    }
}
