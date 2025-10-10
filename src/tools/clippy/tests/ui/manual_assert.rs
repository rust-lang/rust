//@revisions: edition2018 edition2021
//@[edition2018] edition:2018
//@[edition2021] edition:2021

//@no-rustfix: need to change the suggestion to a multipart suggestion

#![warn(clippy::manual_assert)]
#![allow(dead_code, unused_doc_comments)]
#![allow(clippy::nonminimal_bool, clippy::uninlined_format_args, clippy::useless_vec)]

macro_rules! one {
    () => {
        1
    };
}

fn main() {
    let a = vec![1, 2, 3];
    let c = Some(2);
    if !a.is_empty()
        && a.len() == 3
        && c.is_some()
        && !a.is_empty()
        && a.len() == 3
        && !a.is_empty()
        && a.len() == 3
        && !a.is_empty()
        && a.len() == 3
    {
        panic!("qaqaq{:?}", a);
    }
    if !a.is_empty() {
        //~^ manual_assert
        panic!("qaqaq{:?}", a);
    }
    if !a.is_empty() {
        //~^ manual_assert
        panic!("qwqwq");
    }
    if a.len() == 3 {
        println!("qwq");
        println!("qwq");
        println!("qwq");
    }
    if let Some(b) = c {
        panic!("orz {}", b);
    }
    if a.len() == 3 {
        panic!("qaqaq");
    } else {
        println!("qwq");
    }
    let b = vec![1, 2, 3];
    if b.is_empty() {
        //~^ manual_assert
        panic!("panic1");
    }
    if b.is_empty() && a.is_empty() {
        //~^ manual_assert
        panic!("panic2");
    }
    if a.is_empty() && !b.is_empty() {
        //~^ manual_assert
        panic!("panic3");
    }
    if b.is_empty() || a.is_empty() {
        //~^ manual_assert
        panic!("panic4");
    }
    if a.is_empty() || !b.is_empty() {
        //~^ manual_assert
        panic!("panic5");
    }
    if a.is_empty() {
        //~^ manual_assert
        panic!("with expansion {}", one!())
    }
    if a.is_empty() {
        let _ = 0;
    } else if a.len() == 1 {
        panic!("panic6");
    }
}

fn issue7730(a: u8) {
    // Suggestion should preserve comment
    if a > 2 {
        //~^ manual_assert
        // comment
        /* this is a
        multiline
        comment */
        /// Doc comment
        panic!("panic with comment") // comment after `panic!`
    }
}

fn issue12505() {
    struct Foo<T, const N: usize>(T);

    impl<T, const N: usize> Foo<T, N> {
        const BAR: () = if N == 0 {
            //~^ manual_assert
            panic!()
        };
    }
}

fn issue15227(left: u64, right: u64) -> u64 {
    macro_rules! is_x86_feature_detected {
        ($feature:literal) => {
            $feature.len() > 0 && $feature.starts_with("ss")
        };
    }

    if !is_x86_feature_detected!("ssse3") {
        //~^ manual_assert
        panic!("SSSE3 is not supported");
    }
    unsafe { todo!() }
}
