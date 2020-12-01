#[derive(PartialEq, Eq)]
pub struct Tag(pub Context, pub u16);

#[derive(PartialEq, Eq)]
pub enum Context {
    Tiff,
    Exif,
}

impl Tag {
    const ExifIFDPointer: Tag = Tag(Context::Tiff, 34665);
}

fn main() {
    match Tag::ExifIFDPointer {
    //~^ ERROR: non-exhaustive patterns: `Tag(Exif, _)` not covered
        Tag::ExifIFDPointer => {}
    }
}
