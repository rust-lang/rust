/**

An implementation of the Graph500 Bread First Search problem in Rust.

*/

use std;
import std::time;
import std::map;
import std::map::hashmap;
import std::deque;
import std::deque::t;
import std::arc;
import io::writer_util;
import comm::*;
import int::abs;

type node_id = i64;
type graph = [[node_id]];
type bfs_result = [node_id];

iface queue<T: send> {
    fn add_back(T);
    fn pop_front() -> T;
    fn size() -> uint;
}

#[doc="Creates a queue based on ports and channels.

This is admittedly not ideal, but it will help us work around the deque
bugs for the time being."]
fn create_queue<T: send>() -> queue<T> {
    type repr<T: send> = {
        p : port<T>,
        c : chan<T>,
        mut s : uint,
    };

    let p = port();
    let c = chan(p);

    impl<T: copy send> of queue<T> for repr<T> {
        fn add_back(x : T) {
            let x = x;
            send(self.c, x);
            self.s += 1u;
        }

        fn pop_front() -> T {
            self.s -= 1u;
            recv(self.p)
        }

        fn size() -> uint { self.s }
    }

    let Q : repr<T> = { p : p, c : c, mut s : 0u };
    Q as queue::<T>
}

fn make_edges(scale: uint, edgefactor: uint) -> [(node_id, node_id)] {
    let r = rand::rng();

    fn choose_edge(i: node_id, j: node_id, scale: uint, r: rand::rng)
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
            
            let x = r.gen_float();

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

    vec::from_fn((1u << scale) * edgefactor) {|_i|
        choose_edge(0i64, 0i64, scale, r)
    }
}

fn make_graph(N: uint, edges: [(node_id, node_id)]) -> graph {
    let graph = vec::from_fn(N) {|_i| 
        map::hashmap::<node_id, ()>({|x| x as uint }, {|x, y| x == y })
    };

    vec::each(edges) {|e| 
        let (i, j) = e;
        map::set_add(graph[i], j);
        map::set_add(graph[j], i);
        true
    }

    graph.map() {|v|
        map::vec_from_set(v)
    }
}

fn gen_search_keys(graph: graph, n: uint) -> [node_id] {
    let keys = map::hashmap::<node_id, ()>({|x| x as uint }, {|x, y| x == y });
    let r = rand::rng();

    while keys.size() < n {
        let k = r.gen_uint_range(0u, graph.len());

        if graph[k].len() > 0u && vec::any(graph[k]) {|i|
            i != k as node_id
        } {
            map::set_add(keys, k as node_id);
        }
    }
    map::vec_from_set(keys)
}

#[doc="Returns a vector of all the parents in the BFS tree rooted at key.

Nodes that are unreachable have a parent of -1."]
fn bfs(graph: graph, key: node_id) -> bfs_result {
    let marks : [mut node_id] 
        = vec::to_mut(vec::from_elem(vec::len(graph), -1i64));

    let Q = create_queue();

    Q.add_back(key);
    marks[key] = key;

    while Q.size() > 0u {
        let t = Q.pop_front();

        graph[t].each() {|k| 
            if marks[k] == -1i64 {
                marks[k] = t;
                Q.add_back(k);
            }
            true
        };
    }

    vec::from_mut(marks)
}

#[doc="Another version of the bfs function.

This one uses the same algorithm as the parallel one, just without
using the parallel vector operators."]
fn bfs2(graph: graph, key: node_id) -> bfs_result {
    // This works by doing functional updates of a color vector.

    enum color {
        white,
        // node_id marks which node turned this gray/black.
        // the node id later becomes the parent.
        gray(node_id),
        black(node_id)
    };

    let mut colors = vec::from_fn(graph.len()) {|i|
        if i as node_id == key {
            gray(key)
        }
        else {
            white
        }
    };

    fn is_gray(c: color) -> bool {
        alt c {
          gray(_) { true }
          _ { false }
        }
    }

    let mut i = 0u;
    while vec::any(colors, is_gray) {
        // Do the BFS.
        log(info, #fmt("PBFS iteration %?", i));
        i += 1u;
        colors = colors.mapi() {|i, c|
            let c : color = c;
            alt c {
              white {
                let i = i as node_id;
                
                let neighbors = graph[i];
                
                let mut color = white;

                neighbors.each() {|k|
                    if is_gray(colors[k]) {
                        color = gray(k);
                        false
                    }
                    else { true }
                };

                color
              }
              gray(parent) { black(parent) }
              black(parent) { black(parent) }
            }
        }
    }

    // Convert the results.
    vec::map(colors) {|c|
        alt c {
          white { -1i64 }
          black(parent) { parent }
          _ { fail "Found remaining gray nodes in BFS" }
        }
    }
}

#[doc="A parallel version of the bfs function."]
fn pbfs(graph: graph, key: node_id) -> bfs_result {
    // This works by doing functional updates of a color vector.

    enum color {
        white,
        // node_id marks which node turned this gray/black.
        // the node id later becomes the parent.
        gray(node_id),
        black(node_id)
    };

    let mut colors = vec::from_fn(graph.len()) {|i|
        if i as node_id == key {
            gray(key)
        }
        else {
            white
        }
    };

    #[inline(always)]
    fn is_gray(c: color) -> bool {
        alt c {
          gray(_) { true }
          _ { false }
        }
    }

    let (res, graph) = arc::shared_arc(copy graph);

    let mut i = 0u;
    while par::any(colors, is_gray) {
        // Do the BFS.
        log(info, #fmt("PBFS iteration %?", i));
        i += 1u;
        let old_len = colors.len();

        let (res, color) = arc::shared_arc(copy colors);

        colors = par::mapi(colors) {|i, c|
            let c : color = c;
            let colors = &arc::get_arc(color);
            let colors : [color] = *arc::get(colors);
            let graph = &arc::get_arc(graph);
            let graph : graph = *arc::get(graph);
            alt c {
              white {
                let i = i as node_id;
                
                let neighbors = graph[i];
                
                let mut color = white;
                
                neighbors.each() {|k|
                    if is_gray(colors[k]) {
                        color = gray(k);
                        false
                    }
                    else { true }
                    };
                color
              }
              gray(parent) { black(parent) }
              black(parent) { black(parent) }
            }
        };
        assert(colors.len() == old_len);
    }

    // Convert the results.
    par::map(colors) {|c|
        alt c {
          white { -1i64 }
          black(parent) { parent }
          _ { fail "Found remaining gray nodes in BFS" }
        }
    }
}

#[doc="Performs at least some of the validation in the Graph500 spec."]
fn validate(edges: [(node_id, node_id)], 
            root: node_id, tree: bfs_result) -> bool {
    // There are 5 things to test. Below is code for each of them.

    // 1. The BFS tree is a tree and does not contain cycles.
    //
    // We do this by iterating over the tree, and tracing each of the
    // parent chains back to the root. While we do this, we also
    // compute the levels for each node.

    log(info, "Verifying tree structure...");

    let mut status = true;
    let level = tree.map() {|parent| 
        let mut parent = parent;
        let mut path = [];

        if parent == -1i64 {
            // This node was not in the tree.
            -1
        }
        else {
            while parent != root {
                if vec::contains(path, parent) {
                    status = false;
                }

                path += [parent];
                parent = tree[parent];
            }

            // The length of the path back to the root is the current
            // level.
            path.len() as int
        }
    };
    
    if !status { ret status }

    // 2. Each tree edge connects vertices whose BFS levels differ by
    //    exactly one.

    log(info, "Verifying tree edges...");

    let status = tree.alli() {|k, parent|
        if parent != root && parent != -1i64 {
            level[parent] == level[k] - 1
        }
        else {
            true
        }
    };

    if !status { ret status }

    // 3. Every edge in the input list has vertices with levels that
    //    differ by at most one or that both are not in the BFS tree.

    log(info, "Verifying graph edges...");

    let status = edges.all() {|e| 
        let (u, v) = e;

        abs(level[u] - level[v]) <= 1
    };

    if !status { ret status }    

    // 4. The BFS tree spans an entire connected component's vertices.

    // This is harder. We'll skip it for now...

    // 5. A node and its parent are joined by an edge of the original
    //    graph.

    log(info, "Verifying tree and graph edges...");

    let status = par::alli(tree) {|u, v|
        let u = u as node_id;
        if v == -1i64 || u == root {
            true
        }
        else {
            edges.contains((u, v)) || edges.contains((v, u))
        }
    };

    if !status { ret status }    

    // If we get through here, all the tests passed!
    true
}

fn main(args: [str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ["", "15", "48"]
    } else if args.len() <= 1u {
        ["", "10", "16"]
    } else {
        args
    };

    let scale = uint::from_str(args[1]).get();
    let num_keys = uint::from_str(args[2]).get();
    let do_validate = false;
    let do_sequential = true;

    let start = time::precise_time_s();
    let edges = make_edges(scale, 16u);
    let stop = time::precise_time_s();

    io::stdout().write_line(#fmt("Generated %? edges in %? seconds.",
                                 vec::len(edges), stop - start));

    let start = time::precise_time_s();
    let graph = make_graph(1u << scale, edges);
    let stop = time::precise_time_s();

    let mut total_edges = 0u;
    vec::each(graph) {|edges| total_edges += edges.len(); true };

    io::stdout().write_line(#fmt("Generated graph with %? edges in %? seconds.",
                                 total_edges / 2u,
                                 stop - start));

    let mut total_seq = 0.0;
    let mut total_par = 0.0;

    gen_search_keys(graph, num_keys).map() {|root|
        io::stdout().write_line("");
        io::stdout().write_line(#fmt("Search key: %?", root));

        if do_sequential {
            let start = time::precise_time_s();
            let bfs_tree = bfs(graph, root);
            let stop = time::precise_time_s();
            
            //total_seq += stop - start;

            io::stdout().write_line(
                #fmt("Sequential BFS completed in %? seconds.",
                     stop - start));
            
            if do_validate {
                let start = time::precise_time_s();
                assert(validate(edges, root, bfs_tree));
                let stop = time::precise_time_s();
                
                io::stdout().write_line(
                    #fmt("Validation completed in %? seconds.",
                         stop - start));
            }
            
            let start = time::precise_time_s();
            let bfs_tree = bfs2(graph, root);
            let stop = time::precise_time_s();
            
            total_seq += stop - start;
            
            io::stdout().write_line(
                #fmt("Alternate Sequential BFS completed in %? seconds.",
                     stop - start));
            
            if do_validate {
                let start = time::precise_time_s();
                assert(validate(edges, root, bfs_tree));
                let stop = time::precise_time_s();
                
                io::stdout().write_line(
                    #fmt("Validation completed in %? seconds.",
                         stop - start));
            }
        }
        
        let start = time::precise_time_s();
        let bfs_tree = pbfs(graph, root);
        let stop = time::precise_time_s();

        total_par += stop - start;

        io::stdout().write_line(#fmt("Parallel BFS completed in %? seconds.",
                                     stop - start));

        if do_validate {
            let start = time::precise_time_s();
            assert(validate(edges, root, bfs_tree));
            let stop = time::precise_time_s();
            
            io::stdout().write_line(#fmt("Validation completed in %? seconds.",
                                         stop - start));
        }
    };

    io::stdout().write_line("");
    io::stdout().write_line(
        #fmt("Total sequential: %? \t Total Parallel: %? \t Speedup: %?x",
             total_seq, total_par, total_seq / total_par));
}


// par stuff /////////////////////////////////////////////////////////

mod par {
import comm::port;
import comm::chan;
import comm::send;
import comm::recv;
import future::future;

#[doc="The maximum number of tasks this module will spawn for a single
 operationg."]
const max_tasks : uint = 32u;

#[doc="The minimum number of elements each task will process."]
const min_granularity : uint = 1024u;

#[doc="An internal helper to map a function over a large vector and
 return the intermediate results.

This is used to build most of the other parallel vector functions,
like map or alli."]
fn map_slices<A: copy send, B: copy send>(xs: [A],
                                          f: fn~(uint, [const A]/&) -> B) 
    -> [B] {

    let len = xs.len();
    if len < min_granularity {
        log(info, "small slice");
        // This is a small vector, fall back on the normal map.
        [f(0u, xs)]
    }
    else {
        let num_tasks = uint::min(max_tasks, len / min_granularity);

        let items_per_task = len / num_tasks;

        let mut futures = [];
        let mut base = 0u;
        log(info, "spawning tasks");
        while base < len {
            let end = uint::min(len, base + items_per_task);
            // FIXME: why is the ::<A, ()> annotation required here?
            vec::unpack_slice::<A, ()>(xs) {|p, _len|
                let f = ptr::addr_of(f);
                futures += [future::spawn() {|copy base|
                    unsafe {
                        let len = end - base;
                        let slice = (ptr::offset(p, base),
                                     len * sys::size_of::<A>());
                        log(info, #fmt("pre-slice: %?", (base, slice)));
                        let slice : [const A]/& = 
                            unsafe::reinterpret_cast(slice);
                        log(info, #fmt("slice: %?",
                                       (base, vec::len(slice), end - base)));
                        assert(vec::len(slice) == end - base);
                        (*f)(base, slice)
                    }
                }];
            };
            base += items_per_task;
        }
        log(info, "tasks spawned");

        log(info, #fmt("num_tasks: %?", (num_tasks, futures.len())));
        assert(num_tasks == futures.len());

        let r = futures.map() {|ys|
            ys.get()
        };
        assert(r.len() == futures.len());
        r
    }
}

#[doc="A parallel version of map."]
fn map<A: copy send, B: copy send>(xs: [A], f: fn~(A) -> B) -> [B] {
    vec::concat(map_slices(xs) {|_base, slice|
        vec::map(slice, f)
    })
}

#[doc="A parallel version of mapi."]
fn mapi<A: copy send, B: copy send>(xs: [A], f: fn~(uint, A) -> B) -> [B] {
    let slices = map_slices(xs) {|base, slice|
        vec::mapi(slice) {|i, x|
            f(i + base, x)
        }
    };
    let r = vec::concat(slices);
    log(info, (r.len(), xs.len()));
    assert(r.len() == xs.len());
    r
}

#[doc="Returns true if the function holds for all elements in the vector."]
fn alli<A: copy send>(xs: [A], f: fn~(uint, A) -> bool) -> bool {
    vec::all(map_slices(xs) {|base, slice|
        vec::alli(slice) {|i, x|
            f(i + base, x)
        }
    }) {|x| x }
}

    #[doc="Returns true if the function holds for any elements in the vector."]
    fn any<A: copy send>(xs: [A], f: fn~(A) -> bool) -> bool {
        vec::any(map_slices(xs) {|_base, slice|
            vec::any(slice, f)
        }) {|x| x }
    }

}
