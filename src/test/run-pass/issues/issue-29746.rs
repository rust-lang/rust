// run-pass
// zip!(a1,a2,a3,a4) is equivalent to:
//  a1.zip(a2).zip(a3).zip(a4).map(|(((x1,x2),x3),x4)| (x1,x2,x3,x4))
macro_rules! zip {
    // Entry point
    ([$a:expr, $b:expr, $($rest:expr),*]) => {
        zip!([$($rest),*], $a.zip($b), (x,y), [x,y])
    };

    // Intermediate steps to build the zipped expression, the match pattern, and
    //  and the output tuple of the closure, using macro hygiene to repeatedly
    //  introduce new variables named 'x'.
    ([$a:expr, $($rest:expr),*], $zip:expr, $pat:pat, [$($flat:expr),*]) => {
        zip!([$($rest),*], $zip.zip($a), ($pat,x), [$($flat),*, x])
    };

    // Final step
    ([], $zip:expr, $pat:pat, [$($flat:expr),+]) => {
        $zip.map(|$pat| ($($flat),+))
    };

    // Comma
    ([$a:expr], $zip:expr, $pat:pat, [$($flat:expr),*]) => {
        zip!([$a,], $zip, $pat, [$($flat),*])
    };
}

fn main() {
    let p1 = vec![1i32,    2].into_iter();
    let p2 = vec!["10",    "20"].into_iter();
    let p3 = vec![100u16,  200].into_iter();
    let p4 = vec![1000i64, 2000].into_iter();

    let e = zip!([p1,p2,p3,p4]).collect::<Vec<_>>();
    assert_eq!(e[0], (1i32,"10",100u16,1000i64));
}
