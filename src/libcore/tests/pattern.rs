use std::str::pattern::*;

// This macro makes it easier to write
// tests that do a series of iterations
macro_rules! search_asserts {
    ($haystack:expr, $needle:expr, $testname:expr, [$($func:ident),*], $result:expr) => {
        let mut searcher = $needle.into_searcher($haystack);
        let arr = [$( Step::from(searcher.$func()) ),+];
        assert_eq!(&arr[..], &$result, $testname);
    }
}

/// Combined enum for the results of next() and next_match()/next_reject()
#[derive(Debug, PartialEq, Eq)]
enum Step {
    // variant names purposely chosen to
    // be the same length for easy alignment
    Matches(usize, usize),
    Rejects(usize, usize),
    InRange(usize, usize),
    Done
}

use Step::*;

impl From<SearchStep> for Step {
    fn from(x: SearchStep) -> Self {
        match x {
            SearchStep::Match(a, b) => Matches(a, b),
            SearchStep::Reject(a, b) => Rejects(a, b),
            SearchStep::Done => Done
        }
    }
}

impl From<Option<(usize, usize)>> for Step {
    fn from(x: Option<(usize, usize)>) -> Self {
        match x {
            Some((a, b)) => InRange(a, b),
            None => Done
        }
    }
}

#[test]
fn test_simple_iteration() {
    search_asserts! ("abcdeabcd", 'a', "forward iteration for ASCII string",
        // a            b              c              d              e              a              b              c              d              EOF 
        [next,          next,          next,          next,          next,          next,          next,          next,          next,          next],
        [Matches(0, 1), Rejects(1, 2), Rejects(2, 3), Rejects(3, 4), Rejects(4, 5), Matches(5, 6), Rejects(6, 7), Rejects(7, 8), Rejects(8, 9), Done]
    );

    search_asserts! ("abcdeabcd", 'a', "reverse iteration for ASCII string",
        // d            c              b              a            e                d              c              b              a             EOF
        [next_back,     next_back,     next_back,     next_back,     next_back,     next_back,     next_back,     next_back,     next_back,     next_back],
        [Rejects(8, 9), Rejects(7, 8), Rejects(6, 7), Matches(5, 6), Rejects(4, 5), Rejects(3, 4), Rejects(2, 3), Rejects(1, 2), Matches(0, 1), Done]
    );

    search_asserts! ("我爱我的猫", '我', "forward iteration for Chinese string",
        // 我           愛             我             的              貓               EOF
        [next,          next,          next,          next,           next,            next],
        [Matches(0, 3), Rejects(3, 6), Matches(6, 9), Rejects(9, 12), Rejects(12, 15), Done]
    );

    search_asserts! ("我的猫说meow", 'm', "forward iteration for mixed string",
        // 我           的             猫             说              m                e                o                w                EOF
        [next,          next,          next,          next,           next,            next,            next,            next,            next],
        [Rejects(0, 3), Rejects(3, 6), Rejects(6, 9), Rejects(9, 12), Matches(12, 13), Rejects(13, 14), Rejects(14, 15), Rejects(15, 16), Done]
    );

    search_asserts! ("我的猫说meow", '猫', "reverse iteration for mixed string",
        // w             o                 e                m                说              猫             的             我             EOF
        [next_back,       next_back,       next_back,       next_back,       next_back,      next_back,      next_back,    next_back,     next_back],
        [Rejects(15, 16), Rejects(14, 15), Rejects(13, 14), Rejects(12, 13), Rejects(9, 12), Matches(6, 9), Rejects(3, 6), Rejects(0, 3), Done]
    );
}

#[test]
fn test_simple_search() {
    search_asserts!("abcdeabcdeabcde", 'a', "next_match for ASCII string",
        [next_match,    next_match,    next_match,      next_match],
        [InRange(0, 1), InRange(5, 6), InRange(10, 11), Done]
    );

    search_asserts!("abcdeabcdeabcde", 'a', "next_match_back for ASCII string",
        [next_match_back, next_match_back, next_match_back, next_match_back],
        [InRange(10, 11), InRange(5, 6),   InRange(0, 1),   Done]
    );

    search_asserts!("abcdeab", 'a', "next_reject for ASCII string",
        [next_reject,   next_reject,   next_match,    next_reject,   next_reject],
        [InRange(1, 2), InRange(2, 3), InRange(5, 6), InRange(6, 7), Done]
    );

    search_asserts!("abcdeabcdeabcde", 'a', "next_reject_back for ASCII string",
        [next_reject_back, next_reject_back, next_match_back, next_reject_back, next_reject_back, next_reject_back],
        [InRange(14, 15),  InRange(13, 14),  InRange(10, 11), InRange(9, 10),   InRange(8, 9),    InRange(7, 8)]
    );
}

