//@ check-pass

// Unlike `non-root-universe-existential-1.rs` this previously
// compiled as it simply didn't define the hidden type of
// `impl Iterator` when projecting through it. We will do so
// with the new solver. Further minimizing this is challenging.

struct Type(Vec<Type>);
enum TypeTreeValueIter<'a, T> {
    Once(T),
    Ref(&'a ()),
}

impl<'a, T> Iterator for TypeTreeValueIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {}
    }
}

fn item<I: Iterator<Item: Iterator>>(x: I) -> <I::Item as Iterator>::Item {
    loop {}
}

fn get_type_tree_values<'a>(ty: &'a Type) -> impl Iterator<Item = &'a Type> {
    let _: &'a Type = item(std::iter::once(ty).map(get_type_tree_values));
    TypeTreeValueIter::<'a, &'a Type>::Once(ty)
}

fn main() {}
