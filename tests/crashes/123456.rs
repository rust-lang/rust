//@ known-bug: #123456

trait Project {
    const SELF: Self;
}

fn take1(
    _: Project<
        SELF = {
                   j2.join().unwrap();
               },
    >,
) {
}

pub fn main() {}
