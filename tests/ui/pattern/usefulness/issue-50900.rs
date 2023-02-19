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
    //~^ ERROR: match is non-exhaustive
        Tag::ExifIFDPointer => {}
    }
}
