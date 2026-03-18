// https://github.com/rust-lang/rust/issues/4736

struct NonCopyable(());

fn main() {
    let z = NonCopyable{ p: () }; //~ ERROR struct `NonCopyable` has no field named `p`
}
