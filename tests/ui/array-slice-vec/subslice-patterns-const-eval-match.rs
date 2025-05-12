// Test that slice subslice patterns are correctly handled in const evaluation.

//@ run-pass

#[derive(PartialEq, Debug, Clone)]
struct N(u8);

#[derive(PartialEq, Debug, Clone)]
struct Z;

macro_rules! n {
    ($($e:expr),* $(,)?) => {
        [$(N($e)),*]
    }
}

// This macro has an unused variable so that it can be repeated base on the
// number of times a repeated variable (`$e` in `z`) occurs.
macro_rules! zed {
    ($e:expr) => { Z }
}

macro_rules! z {
    ($($e:expr),* $(,)?) => {
        [$(zed!($e)),*]
    }
}

// Compare constant evaluation and runtime evaluation of a given expression.
macro_rules! compare_evaluation_inner {
    ($e:expr, $t:ty $(,)?) => {{
        const CONST_EVAL: $t = $e;
        const fn const_eval() -> $t { $e }
        static CONST_EVAL2: $t = const_eval();
        let runtime_eval = $e;
        assert_eq!(CONST_EVAL, runtime_eval);
        assert_eq!(CONST_EVAL2, runtime_eval);
    }}
}

// Compare the result of matching `$e` against `$p` using both `if let` and
// `match`.
macro_rules! compare_evaluation {
    ($p:pat, $e:expr, $matches:expr, $t:ty $(,)?) => {{
        compare_evaluation_inner!(if let $p = $e as &[_] { $matches } else { None }, $t);
        compare_evaluation_inner!(match $e as &[_] { $p => $matches, _ => None }, $t);
    }}
}

// Repeat `$test`, substituting the given macro variables with the given
// identifiers.
//
// For example:
//
// repeat! {
//     ($name); X; Y:
//     struct $name;
// }
//
// Expands to:
//
// struct X; struct Y;
//
// This is used to repeat the tests using both the `N` and `Z`
// types.
macro_rules! repeat {
    (($($dollar:tt $placeholder:ident)*); $($($values:ident),+);*: $($test:tt)*) => {
        macro_rules! single {
            ($($dollar $placeholder:ident),*) => { $($test)* }
        }
        $(single!($($values),+);)*
    }
}

fn main() {
    repeat! {
        ($arr $Ty); n, N; z, Z:
        compare_evaluation!([_, x @ .., _], &$arr!(1, 2, 3, 4), Some(x), Option<&'static [$Ty]>);
        compare_evaluation!([x, .., _], &$arr!(1, 2, 3, 4), Some(x), Option<&'static $Ty>);
        compare_evaluation!([_, .., x], &$arr!(1, 2, 3, 4), Some(x), Option<&'static $Ty>);

        compare_evaluation!([_, x @ .., _], &$arr!(1, 2), Some(x), Option<&'static [$Ty]>);
        compare_evaluation!([x, .., _], &$arr!(1, 2), Some(x), Option<&'static $Ty>);
        compare_evaluation!([_, .., x], &$arr!(1, 2), Some(x), Option<&'static $Ty>);

        compare_evaluation!([_, x @ .., _], &$arr!(1), Some(x), Option<&'static [$Ty]>);
        compare_evaluation!([x, .., _], &$arr!(1), Some(x), Option<&'static $Ty>);
        compare_evaluation!([_, .., x], &$arr!(1), Some(x), Option<&'static $Ty>);
    }

    compare_evaluation!([N(x), .., _], &n!(1, 2, 3, 4), Some(x), Option<&'static u8>);
    compare_evaluation!([_, .., N(x)], &n!(1, 2, 3, 4), Some(x), Option<&'static u8>);

    compare_evaluation!([N(x), .., _], &n!(1, 2), Some(x), Option<&'static u8>);
    compare_evaluation!([_, .., N(x)], &n!(1, 2), Some(x), Option<&'static u8>);
}
