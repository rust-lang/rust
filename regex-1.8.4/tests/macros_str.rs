// Macros for use in writing tests generic over &str/&[u8].
macro_rules! text { ($text:expr) => { $text } }
macro_rules! t { ($text:expr) => { text!($text) } }
macro_rules! match_text { ($text:expr) => { $text.as_str() } }
macro_rules! use_ { ($($path: tt)*) => { use regex::$($path)*; } }
macro_rules! empty_vec { () => { <Vec<&str>>::new() } }
macro_rules! bytes { ($text:expr) => { std::str::from_utf8($text.as_ref()).unwrap() } }

macro_rules! no_expand {
    ($text:expr) => {{
        use regex::NoExpand;
        NoExpand(text!($text))
    }}
}

macro_rules! show { ($text:expr) => { $text } }

// N.B. The expansion API for &str and &[u8] APIs differs slightly for now,
// but they should be unified in 1.0. Then we can move this macro back into
// tests/api.rs where it is used. ---AG
macro_rules! expand {
    ($name:ident, $re:expr, $text:expr, $expand:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let re = regex!($re);
            let cap = re.captures(t!($text)).unwrap();

            let mut got = String::new();
            cap.expand(t!($expand), &mut got);
            assert_eq!(show!(t!($expected)), show!(&*got));
        }
    }
}

#[cfg(feature = "pattern")]
macro_rules! searcher_expr { ($e:expr) => ($e) }
#[cfg(not(feature = "pattern"))]
macro_rules! searcher_expr { ($e:expr) => ({}) }
