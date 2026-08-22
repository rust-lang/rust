#![feature(macro_guard_matcher)]

fn main() {
    macro_rules! m {
        ($g:guard) => {};
    }

    // Accepts
    m!(if true);
    m!(if let Some(x) = Some(1));
    m!(if let Some(x) = Some(1) && x == 1);
    m!(if let Some(x) = Some(Some(1)) && let Some(1) = x);
    m!(if let Some(x) = Some(Some(1)) && let Some(y) = x && y == 1);

    // Rejects
    m!(let Some(x) = Some(1)); //~ERROR no rules expected keyword `let`

    macro_rules! m_m {
        (case1, $g:guard) => {
            m!($g)
        };
        (case2, if $e:expr) => {
            m!(if $e)
        };
    }

    // Accepted since `m` recognizes that the sequence produced by the expansion of
    // metavar `$g` "begins" (i.e., is) a guard since it's of kind `guard`.
    m_m!(case1, if true);
    m_m!(case2, if true);
}
