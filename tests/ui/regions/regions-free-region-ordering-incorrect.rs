// Test that free regions ordering only goes one way. That is,
// we have `&'a Node<'b, T>`, which implies that `'a <= 'b`,
// but not `'b <= 'a`. Hence, returning `&self.val` (which has lifetime
// `'a`) where `'b` is expected yields an error.
//
// This test began its life as a test for issue #4325.

struct Node<'b, T: 'b> {
    val: T,
    next: Option<&'b Node<'b, T>>
}

impl<'b, T> Node<'b, T> {
    fn get<'a>(&'a self) -> &'b T {
        match self.next { //~ ERROR lifetime may not live long enough
            Some(ref next) => next.get(),
            None => &self.val
        }
    }
}

fn main() {}
