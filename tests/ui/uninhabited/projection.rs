// check-pass

#![feature(never_type, exhaustive_patterns)]

trait Tag {
    type TagType;
}

enum Keep {}
enum Erase {}

impl Tag for Keep {
    type TagType = ();
}

impl Tag for Erase {
    type TagType = !;
}

enum TagInt<T: Tag> {
    Untagged(i32),
    Tagged(T::TagType, i32)
}

fn test(keep: TagInt<Keep>, erase: TagInt<Erase>) {
    match erase {
        TagInt::Untagged(_) => (),
        TagInt::Tagged(_, _) => ()
    };
}

fn main() {}
