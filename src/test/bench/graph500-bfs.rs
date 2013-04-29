// xfail-pretty

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(deprecated_mode)];

/*!

An implementation of the Graph500 Breadth First Search problem in Rust.

*/

extern mod std;
use std::arc;
use std::time;
use std::deque::Deque;
use std::par;
use core::hashmap::{HashMap, HashSet};
use core::int::abs;
use core::rand::RngUtil;

type node_id = i64;
type graph = ~[~[node_id]];
type bfs_result = ~[node_id];

fn make_edges(scale: uint, edgefactor: uint) -> ~[(node_id, node_id)] {
    let r = rand::XorShiftRng::new();

    fn choose_edge<R: rand::Rng>(i: node_id, j: node_id, scale: uint, r: &R)
        -> (node_id, node_id) {

        let A = 0.57;
        let B = 0.19;
        let C = 0.19;

        if scale == 0u {
            (i, j)
        }
        else {
            let i = i * 2i64;
            let j = j * 2i64;
            let scale = scale - 1u;

            let x = r.gen::<float>();

            if x < A {
                choose_edge(i, j, scale, r)
            }
            else {
                let x = x - A;
                if x < B {
                    choose_edge(i + 1i64, j, scale, r)
                }
                else {
                    let x = x - B;
                    if x < C {
                        choose_edge(i, j + 1i64, scale, r)
                    }
                    else {
                        choose_edge(i + 1i64, j + 1i64, scale, r)
                    }
                }
            }
        }
    }

    do vec::from_fn((1u << scale) * edgefactor) |_i| {
        choose_edge(0i64, 0i64, scale, &r)
    }
}

fn make_graph(N: uint, edges: ~[(node_id, node_id)]) -> graph {
    let mut graph = do vec::from_fn(N) |_i| {
        HashSet::new()
    };

    do vec::each(edges) |e| {
        match *e {
            (i, j) => {
                graph[i].insert(j);
                graph[j].insert(i);
            }
        }
        true
    }

    do vec::map_consume(graph) |mut v| {
        let mut vec = ~[];
        do v.consume |i| {
            vec.push(i);
        }
        vec
    }
}

fn gen_search_keys(graph: &[~[node_id]], n: uint) -> ~[node_id] {
    let mut keys = HashSet::new();
    let r = rand::rng();

    while keys.len() < n {
        let k = r.gen_uint_range(0u, graph.len());

        if graph[k].len() > 0u && vec::any(graph[k], |i| {
            *i != k as node_id
        }) {
            keys.insert(k as node_id);
        }
    }
    let mut vec = ~[];
    do keys.consume |i| {
        vec.push(i);
    }
    return vec;
}

/**
 * Returns a vector of all the parents in the BFS tree rooted at key.
 *
 * Nodes that are unreachable have a parent of -1.
 */
fn bfs(graph: graph, key: node_id) -> bfs_result {
    let mut marks : ~[node_id]
        = vec::from_elem(vec::len(graph), -1i64);

    let mut q = Deque::new();

    q.add_back(key);
    marks[key] = key;

    while !q.is_empty() {
        let t = q.pop_front();

        do graph[t].each() |k| {
            if marks[*k] == -1i64 {
                marks[*k] = t;
                q.add_back(*k);
            }
            true
        };
    }

    marks
}

/**
 * Another version of the bfs function.
 *
 * This one uses the same algorithm as the parallel one, just without
 * using the parallel vector operators.
 */
fn bfs2(graph: graph, key: node_id) -> bfs_result {
    // This works by doing functional updates of a color vector.

    enum color {
        white,
        // node_id marks which node turned this gray/black.
        // the node id later becomes the parent.
        gray(node_id),
        black(node_id)
    };

    let mut colors = do vec::from_fn(graph.len()) |i| {
        if i as node_id == key {
            gray(key)
        }
        else {
            white
        }
    };

    fn is_gray(c: &color) -> bool {
        match *c {
          gray(_) => { true }
          _ => { false }
        }
    }

    let mut i = 0;
    while vec::any(colors, is_gray) {
        // Do the BFS.
        info!("PBFS iteration %?", i);
        i += 1;
        colors = do colors.mapi() |i, c| {
            let c : color = *c;
            match c {
              white => {
                let i = i as node_id;

                let neighbors = copy graph[i];

                let mut color = white;

                do neighbors.each() |k| {
                    if is_gray(&colors[*k]) {
                        color = gray(*k);
                        false
                    }
                    else { true }
                };

                color
              }
              gray(parent) => { black(parent) }
              black(parent) => { black(parent) }
            }
        }
    }

    // Convert the results.
    do vec::map(colors) |c| {
        match *c {
          white => { -1i64 }
          black(parent) => { parent }
          _ => { fail!(~"Found remaining gray nodes in BFS") }
        }
    }
}

/// A parallel version of the bfs function.
fn pbfs(graph: &arc::ARC<graph>, key: node_id) -> bfs_result {
    // This works by doing functional updates of a color vector.

    enum color {
        white,
        // node_id marks which node turned this gray/black.
        // the node id later becomes the parent.
        gray(node_id),
        black(node_id)
    };

    let graph_vec = arc::get(graph); // FIXME #3387 requires this temp
    let mut colors = do vec::from_fn(graph_vec.len()) |i| {
        if i as node_id == key {
            gray(key)
        }
        else {
            white
        }
    };

    #[inline(always)]
    fn is_gray(c: &color) -> bool {
        match *c {
          gray(_) => { true }
          _ => { false }
        }
    }

    fn is_gray_factory() -> ~fn(c: &color) -> bool {
        let r: ~fn(c: &color) -> bool = is_gray;
        r
    }

    let mut i = 0;
    while par::any(colors, is_gray_factory) {
        // Do the BFS.
        info!("PBFS iteration %?", i);
        i += 1;
        let old_len = colors.len();

        let color = arc::ARC(colors);

        let color_vec = arc::get(&color); // FIXME #3387 requires this temp
        colors = do par::mapi(*color_vec) {
            let colors = arc::clone(&color);
            let graph = arc::clone(graph);
            let result: ~fn(+x: uint, +y: &color) -> color = |i, c| {
                let colors = arc::get(&colors);
                let graph = arc::get(&graph);
                match *c {
                  white => {
                    let i = i as node_id;

                    let neighbors = copy graph[i];

                    let mut color = white;

                    do neighbors.each() |k| {
                        if is_gray(&colors[*k]) {
                            color = gray(*k);
                            false
                        }
                        else { true }
                    };
                    color
                  }
                  gray(parent) => { black(parent) }
                  black(parent) => { black(parent) }
                }
            };
            result
        };
        assert!((colors.len() == old_len));
    }

    // Convert the results.
    do par::map(colors) {
        let result: ~fn(c: &color) -> i64 = |c| {
            match *c {
                white => { -1i64 }
                black(parent) => { parent }
                _ => { fail!(~"Found remaining gray nodes in BFS") }
            }
        };
        result
    }
}

/// Performs at least some of the validation in the Graph500 spec.
fn validate(edges: ~[(node_id, node_id)],
            root: node_id, tree: bfs_result) -> bool {
    // There are 5 things to test. Below is code for each of them.

    // 1. The BFS tree is a tree and does not contain cycles.
    //
    // We do this by iterating over the tree, and tracing each of the
    // parent chains back to the root. While we do this, we also
    // compute the levels for each node.

    info!(~"Verifying tree structure...");

    let mut status = true;
    let level = do tree.map() |parent| {
        let mut parent = *parent;
        let mut path = ~[];

        if parent == -1i64 {
            // This node was not in the tree.
            -1
        }
        else {
            while parent != root {
                if vec::contains(path, &parent) {
                    status = false;
                }

                path.push(parent);
                parent = tree[parent];
            }

            // The length of the path back to the root is the current
            // level.
            path.len() as int
        }
    };

    if !status { return status }

    // 2. Each tree edge connects vertices whose BFS levels differ by
    //    exactly one.

    info!(~"Verifying tree edges...");

    let status = do tree.alli() |k, parent| {
        if *parent != root && *parent != -1i64 {
            level[*parent] == level[k] - 1
        }
        else {
            true
        }
    };

    if !status { return status }

    // 3. Every edge in the input list has vertices with levels that
    //    differ by at most one or that both are not in the BFS tree.

    info!(~"Verifying graph edges...");

    let status = do edges.all() |e| {
        let (u, v) = *e;

        abs(level[u] - level[v]) <= 1
    };

    if !status { return status }

    // 4. The BFS tree spans an entire connected component's vertices.

    // This is harder. We'll skip it for now...

    // 5. A node and its parent are joined by an edge of the original
    //    graph.

    info!(~"Verifying tree and graph edges...");

    let status = do par::alli(tree) {
        let edges = copy edges;
        let result: ~fn(+x: uint, v: &i64) -> bool = |u, v| {
            let u = u as node_id;
            if *v == -1i64 || u == root {
                true
            } else {
                edges.contains(&(u, *v)) || edges.contains(&(*v, u))
            }
        };
        result
    };

    if !status { return status }

    // If we get through here, all the tests passed!
    true
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"15", ~"48"]
    } else if args.len() <= 1 {
        ~[~"", ~"10", ~"16"]
    } else {
        args
    };

    let scale = uint::from_str(args[1]).get();
    let num_keys = uint::from_str(args[2]).get();
    let do_validate = false;
    let do_sequential = true;

    let start = time::precise_time_s();
    let edges = make_edges(scale, 16);
    let stop = time::precise_time_s();

    io::stdout().write_line(fmt!("Generated %? edges in %? seconds.",
                                 vec::len(edges), stop - start));

    let start = time::precise_time_s();
    let graph = make_graph(1 << scale, copy edges);
    let stop = time::precise_time_s();

    let mut total_edges = 0;
    vec::each(graph, |edges| { total_edges += edges.len(); true });

    io::stdout().write_line(fmt!("Generated graph with %? edges in %? seconds.",
                                 total_edges / 2,
                                 stop - start));

    let mut total_seq = 0.0;
    let mut total_par = 0.0;

    let graph_arc = arc::ARC(copy graph);

    do gen_search_keys(graph, num_keys).map() |root| {
        io::stdout().write_line(~"");
        io::stdout().write_line(fmt!("Search key: %?", root));

        if do_sequential {
            let start = time::precise_time_s();
            let bfs_tree = bfs(copy graph, *root);
            let stop = time::precise_time_s();

            //total_seq += stop - start;

            io::stdout().write_line(
                fmt!("Sequential BFS completed in %? seconds.",
                     stop - start));

            if do_validate {
                let start = time::precise_time_s();
                assert!((validate(copy edges, *root, bfs_tree)));
                let stop = time::precise_time_s();

                io::stdout().write_line(
                    fmt!("Validation completed in %? seconds.",
                         stop - start));
            }

            let start = time::precise_time_s();
            let bfs_tree = bfs2(copy graph, *root);
            let stop = time::precise_time_s();

            total_seq += stop - start;

            io::stdout().write_line(
                fmt!("Alternate Sequential BFS completed in %? seconds.",
                     stop - start));

            if do_validate {
                let start = time::precise_time_s();
                assert!((validate(copy edges, *root, bfs_tree)));
                let stop = time::precise_time_s();

                io::stdout().write_line(
                    fmt!("Validation completed in %? seconds.",
                         stop - start));
            }
        }

        let start = time::precise_time_s();
        let bfs_tree = pbfs(&graph_arc, *root);
        let stop = time::precise_time_s();

        total_par += stop - start;

        io::stdout().write_line(fmt!("Parallel BFS completed in %? seconds.",
                                     stop - start));

        if do_validate {
            let start = time::precise_time_s();
            assert!((validate(copy edges, *root, bfs_tree)));
            let stop = time::precise_time_s();

            io::stdout().write_line(fmt!("Validation completed in %? seconds.",
                                         stop - start));
        }
    };

    io::stdout().write_line(~"");
    io::stdout().write_line(
        fmt!("Total sequential: %? \t Total Parallel: %? \t Speedup: %?x",
             total_seq, total_par, total_seq / total_par));
}
