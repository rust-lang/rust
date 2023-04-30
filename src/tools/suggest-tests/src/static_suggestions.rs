use crate::{sug, Suggestion};

// FIXME: perhaps this could use `std::lazy` when it is stablizied
macro_rules! static_suggestions {
    ($( $glob:expr => [ $( $suggestion:expr ),* ] ),*) => {
        pub(crate) const STATIC_SUGGESTIONS: ::once_cell::unsync::Lazy<Vec<(&'static str, Vec<Suggestion>)>>
            = ::once_cell::unsync::Lazy::new(|| vec![ $( ($glob, vec![ $($suggestion),* ]) ),*]);
    }
}

static_suggestions! {
    "*.md" => [
        sug!("test", 0, ["linkchecker"])
    ],

    "compiler/*" => [
        sug!("check"),
        sug!("test", 1, ["tests/ui", "tests/run-make"])
    ],

    "src/librustdoc/*" => [
        sug!("test", 1, ["rustdoc"])
    ]
}
