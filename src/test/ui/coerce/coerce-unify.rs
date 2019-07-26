// run-pass
// Check that coercions can unify if-else, match arms and array elements.

// Try to construct if-else chains, matches and arrays out of given expressions.
macro_rules! check {
    ($last:expr $(, $rest:expr)+) => {
        // Last expression comes first because of whacky ifs and matches.
        let _ = $(if false { $rest })else+ else { $last };

        let _ = match 0 { $(_ if false => $rest,)+ _ => $last };

        let _ = [$($rest,)+ $last];
    }
}

// Check all non-uniform cases of 2 and 3 expressions of 2 types.
macro_rules! check2 {
    ($a:expr, $b:expr) => {
        check!($a, $b);
        check!($b, $a);

        check!($a, $a, $b);
        check!($a, $b, $a);
        check!($a, $b, $b);

        check!($b, $a, $a);
        check!($b, $a, $b);
        check!($b, $b, $a);
    }
}

// Check all non-uniform cases of 2 and 3 expressions of 3 types.
macro_rules! check3 {
    ($a:expr, $b:expr, $c:expr) => {
        // Delegate to check2 for cases where a type repeats.
        check2!($a, $b);
        check2!($b, $c);
        check2!($a, $c);

        // Check the remaining cases, i.e., permutations of ($a, $b, $c).
        check!($a, $b, $c);
        check!($a, $c, $b);
        check!($b, $a, $c);
        check!($b, $c, $a);
        check!($c, $a, $b);
        check!($c, $b, $a);
    }
}

use std::mem::size_of;

fn foo() {}
fn bar() {}

pub fn main() {
    check3!(foo, bar, foo as fn());
    check3!(size_of::<u8>, size_of::<u16>, size_of::<usize> as fn() -> usize);

    let s = String::from("bar");
    check2!("foo", &s);

    let a = [1, 2, 3];
    let v = vec![1, 2, 3];
    check2!(&a[..], &v);

    // Make sure in-array coercion still works.
    let _ = [("a", Default::default()), (Default::default(), "b"), (&s, &s)];
}
