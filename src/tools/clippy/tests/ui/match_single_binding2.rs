#![warn(clippy::match_single_binding)]
#![allow(unused_variables)]
#![allow(clippy::uninlined_format_args)]

fn main() {
    // Lint (additional curly braces needed, see #6572)
    struct AppendIter<I>
    where
        I: Iterator,
    {
        inner: Option<(I, <I as Iterator>::Item)>,
    }

    #[allow(dead_code)]
    fn size_hint<I: Iterator>(iter: &AppendIter<I>) -> (usize, Option<usize>) {
        match &iter.inner {
            Some((iter, _item)) => match iter.size_hint() {
                (min, max) => (min.saturating_add(1), max.and_then(|max| max.checked_add(1))),
            },
            None => (0, Some(0)),
        }
    }

    // Lint (no additional curly braces needed)
    let opt = Some((5, 2));
    let get_tup = || -> (i32, i32) { (1, 2) };
    match opt {
        #[rustfmt::skip]
        Some((first, _second)) => {
            match get_tup() {
                (a, b) => println!("a {:?} and b {:?}", a, b),
            }
        },
        None => println!("nothing"),
    }

    fn side_effects() {}

    // Lint (scrutinee has side effects)
    // issue #7094
    match side_effects() {
        _ => println!("Side effects"),
    }

    // Lint (scrutinee has side effects)
    // issue #7094
    let x = 1;
    match match x {
        0 => 1,
        _ => 2,
    } {
        _ => println!("Single branch"),
    }
}
