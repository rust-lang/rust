// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A test of the macro system. Can we do HTML literals?

#![feature(macro_rules)]


/*

This is an HTML parser written as a macro. It's all CPS, and we have
to carry around a bunch of state. The arguments to macros all look like this:

{ tag_stack* # expr* # tokens }

The stack keeps track of where we are in the tree. The expr is a list
of children of the current node. The tokens are everything that's
left.

*/

macro_rules! html (
    ( $($body:tt)* ) => (
        parse_node!( []; []; $($body)* )
    )
)

macro_rules! parse_node (
    (
        [:$head:ident ($(:$head_nodes:expr),*)
         $(:$tags:ident ($(:$tag_nodes:expr),*))*];
        [$(:$nodes:expr),*];
        </$tag:ident> $($rest:tt)*
    ) => (
        parse_node!(
            [$(: $tags ($(:$tag_nodes),*))*];
            [$(:$head_nodes,)* :tag(stringify!($head).to_owned(),
                                    vec!($($nodes),*))];
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
            [$(:$nodes,)* :text(~".")];
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
            [$(:$nodes,)* :text(stringify!($word).to_owned())];
            $($rest)*
        )
    );

    ( []; [:$e:expr]; ) => ( $e );
)

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

enum HTMLFragment {
    tag(~str, Vec<HTMLFragment> ),
    text(~str),
}
