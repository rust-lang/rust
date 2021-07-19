use std::{borrow::Cow, collections::BTreeMap, marker::PhantomData, sync::Arc};

fn byte_view<'a>(s: &'a ByteView<'_>) -> BTreeMap<&'a str, ByteView<'a>> {
    panic!()
}

fn group_entries(s: &()) -> BTreeMap<Cow<'_, str>, Vec<Cow<'_, str>>> {
    todo!()
}

struct Mmap;

enum ByteViewBacking<'a> {
    Buf(Cow<'a, [u8]>),
    Mmap(Mmap),
}

pub struct ByteView<'a> {
    backing: Arc<ByteViewBacking<'a>>,
}

fn main() {
    byte_view(panic!());
    group_entries(panic!());
}
