// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! graph {
    ($( $source:ident -> $target:ident, )*) => {
        {
            use $crate::rustc_data_structures::graph::{Graph, NodeIndex};
            use $crate::rustc_data_structures::fx::FxHashMap;

            let mut graph = Graph::new();
            let mut nodes: FxHashMap<&'static str, NodeIndex> = FxHashMap();

            for &name in &[ $(stringify!($source), stringify!($target)),* ] {
                let name: &'static str = name;
                nodes.entry(name)
                     .or_insert_with(|| graph.add_node(name));
            }

            $(
                {
                    let source = nodes[&stringify!($source)];
                    let target = nodes[&stringify!($target)];
                    graph.add_edge(source, target, ());
                }
            )*

            let f = move |name: &'static str| -> NodeIndex { nodes[&name] };

            (graph, f)
        }
    }
}

macro_rules! set {
    ($( $value:expr ),*) => {
        {
            use $crate::rustc_data_structures::fx::FxHashSet;
            let mut set = FxHashSet();
            $(set.insert($value);)*
            set
        }
    }
}
