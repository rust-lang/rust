#![feature(explicit_tail_calls)]

pub const fn test(_: &View) {
    const fn takes_view(_: &View) {}

    become takes_view(HasDrop.as_view());
    //~^ error: temporary value dropped while borrowed
}

struct HasDrop;
struct View;

impl HasDrop {
    const fn as_view(&self) -> &View {
        &View
    }
}

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {}
