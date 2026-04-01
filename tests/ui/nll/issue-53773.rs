struct Archive;
struct ArchiveIterator<'a> {
    x: &'a Archive,
}
struct ArchiveChild<'a> {
    x: &'a Archive,
}

struct A {
    raw: &'static mut Archive,
}
struct Iter<'a> {
    raw: &'a mut ArchiveIterator<'a>,
}
struct C<'a> {
    raw: &'a mut ArchiveChild<'a>,
}

impl A {
    pub fn iter(&self) -> Iter<'_> {
        panic!()
    }
}
impl Drop for A {
    fn drop(&mut self) {}
}
impl<'a> Drop for C<'a> {
    fn drop(&mut self) {}
}

impl<'a> Iterator for Iter<'a> {
    type Item = C<'a>;
    fn next(&mut self) -> Option<C<'a>> {
        panic!()
    }
}

fn error(archive: &A) {
    let mut members: Vec<&mut ArchiveChild<'_>> = vec![];
    for child in archive.iter() {
        members.push(child.raw);
        //~^ ERROR borrow may still be in use when destructor runs [E0713]
    }
    members.len();
}

fn main() {}
