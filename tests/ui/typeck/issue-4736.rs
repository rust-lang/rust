struct NonCopyable(());

fn main() {
    let z = NonCopyable{ p: () }; //~ ERROR struct `NonCopyable` has no field named `p`
}
