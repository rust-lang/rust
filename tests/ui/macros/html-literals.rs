//@ run-pass

#![allow(non_camel_case_types)]
// A test of the macro system. Can we do HTML literals?

/*

This is an HTML parser written as a macro. It's all CPS, and we have
to carry around a bunch of state. The arguments to macros all look like this:

{ tag_stack* # expr* # tokens }

The stack keeps track of where we are in the tree. The expr is a list
of children of the current node. The tokens are everything that's
left.

*/
use HTMLFragment::{tag, text};

macro_rules! html {
    ( $($body:tt)* ) => (
        parse_node!( []; []; $($body)* )
    )
}

macro_rules! parse_node {
    (
        [:$head:ident ($(:$head_nodes:expr),*)
         $(:$tags:ident ($(:$tag_nodes:expr),*))*];
        [$(:$nodes:expr),*];
        </$tag:ident> $($rest:tt)*
    ) => (
        parse_node!(
            [$(: $tags ($(:$tag_nodes),*))*];
            [$(:$head_nodes,)* :tag(stringify!($head).to_string(),
                                    vec![$($nodes),*])];
            $($rest)*
        )
    );

    (
        [$(:$tags:ident ($(:$tag_nodes:expr),*) )*];
        [$(:$nodes:expr),*];
        <$tag:ident> $($rest:tt)*
    ) => (
        parse_node!(
            [:$tag ($(:$nodes)*) $(: $tags ($(:$tag_nodes),*) )*];
            [];
            $($rest)*
        )
    );

    (
        [$(:$tags:ident ($(:$tag_nodes:expr),*) )*];
        [$(:$nodes:expr),*];
        . $($rest:tt)*
    ) => (
        parse_node!(
            [$(: $tags ($(:$tag_nodes),*))*];
            [$(:$nodes,)* :text(".".to_string())];
            $($rest)*
        )
    );

    (
        [$(:$tags:ident ($(:$tag_nodes:expr),*) )*];
        [$(:$nodes:expr),*];
        $word:ident $($rest:tt)*
    ) => (
        parse_node!(
            [$(: $tags ($(:$tag_nodes),*))*];
            [$(:$nodes,)* :text(stringify!($word).to_string())];
            $($rest)*
        )
    );

    ( []; [:$e:expr]; ) => ( $e );
}

pub fn main() {
    let _page = html! (
        <html>
            <head><title>This is the title.</title></head>
            <body>
            <p>This is some text</p>
            </body>
        </html>
    );
}

#[allow(dead_code)]
enum HTMLFragment {
    tag(String, Vec<HTMLFragment> ),
    text(String),
}
