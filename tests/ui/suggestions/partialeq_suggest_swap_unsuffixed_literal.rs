// The operands of `==` can be swapped when only the reversed `PartialEq` impl exists. Check
// that this is still noticed when unsuffixed literals leave `{integer}` in the operand types,
// and that a comparison inside of an external macro gets the note without a suggestion.

struct T(i32);

impl PartialEq<i32> for T {
    fn eq(&self, other: &i32) -> bool {
        &self.0 == other
    }
}

struct W<A>(A);

impl<A: PartialEq<B>, B> PartialEq<Option<B>> for W<A> {
    fn eq(&self, other: &Option<B>) -> bool {
        other.as_ref().is_some_and(|other| &self.0 == other)
    }
}

fn main() {
    let v = vec![1, 2, 3];
    let a = [1, 2, 3];
    let _ = a == v; //~ ERROR can't compare `[{integer}; 3]` with `Vec<{integer}>` [E0277]
    let _ = a != v; //~ ERROR can't compare `[{integer}; 3]` with `Vec<{integer}>` [E0277]
    assert_eq!(a, v); //~ ERROR can't compare `[{integer}; 3]` with `Vec<{integer}>` [E0277]
    let _ = 4 == T(4); //~ ERROR can't compare `{integer}` with `T` [E0277]
    let _ = Some(1) == W(1); //~ ERROR mismatched types [E0308]
    // `T` can't be compared with a float, so there is nothing to swap.
    let _ = 1.5 == T(4); //~ ERROR can't compare `{float}` with `T` [E0277]
}
