// revisions: edition2018 edition2021
//[edition2018] edition:2018
//[edition2021] edition:2021
// run-rustfix

#![warn(clippy::manual_assert)]
#![allow(dead_code, unused_doc_comments)]
#![allow(clippy::nonminimal_bool, clippy::uninlined_format_args)]

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
        panic!("qaqaq{:?}", a);
    }
    if !a.is_empty() {
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
        panic!("panic1");
    }
    if b.is_empty() && a.is_empty() {
        panic!("panic2");
    }
    if a.is_empty() && !b.is_empty() {
        panic!("panic3");
    }
    if b.is_empty() || a.is_empty() {
        panic!("panic4");
    }
    if a.is_empty() || !b.is_empty() {
        panic!("panic5");
    }
    if a.is_empty() {
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
        // comment
        /* this is a
        multiline
        comment */
        /// Doc comment
        panic!("panic with comment") // comment after `panic!`
    }
}
