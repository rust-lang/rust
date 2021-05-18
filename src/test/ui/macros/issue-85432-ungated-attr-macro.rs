// check-pass
// Regression test for issue #85432
// Ensures that we don't incorrectly gate nonterminals
// in key-value macros when we need to reparse due to
// the presence of `#[derive]`

macro_rules! with_doc_comment {
    ($comment:expr, $item:item) => {
        #[doc = $comment]
        $item
    };
}

macro_rules! database_table_doc {
    () => {
        ""
    };
}

with_doc_comment! {
    database_table_doc!(),
    #[derive(Debug)]
    struct Image {
        #[cfg(FALSE)]
        _f: (),
    }

}

fn main() {}
