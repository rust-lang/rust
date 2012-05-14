/**

An implementation of the Graph500 Bread First Search problem in Rust.

*/

use std;
import std::time;
import std::map;
import std::map::hashmap;
import std::deque;
import std::deque::t;
import io::writer_util;
import vec::extensions;
import comm::*;

type node_id = i64;
type graph = [map::set<node_id>];

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

    impl<T: send> of queue<T> for repr<T> {
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
            let i = i * 2;
            let j = j * 2;
            let scale = scale - 1u;
            
            let x = r.next_float();

            if x < A {
                choose_edge(i, j, scale, r)
            }
            else {
                let x = x - A;
                if x < B {
                    choose_edge(i + 1, j, scale, r)
                }
                else {
                    let x = x - B;
                    if x < C {
                        choose_edge(i, j + 1, scale, r)
                    }
                    else {
                        choose_edge(i + 1, j + 1, scale, r)
                    }
                }
            }
        }
    }

    vec::from_fn((1u << scale) * edgefactor) {|_i|
        choose_edge(0, 0, scale, r)
    }
}

fn make_graph(N: uint, edges: [(node_id, node_id)]) -> graph {
    let graph = vec::from_fn(N) {|_i| map::int_hash() };

    vec::each(edges) {|e| 
        let (i, j) = e;
        map::set_add(graph[i], j);
        map::set_add(graph[j], i);
        true
    }

    graph
}

#[doc="Returns a vector of all the parents in the BFS tree rooted at key.

Nodes that are unreachable have a parent of -1."]
fn bfs(graph: graph, key: node_id) -> [node_id] {
    let marks : [mut node_id] 
        = vec::to_mut(vec::from_elem(vec::len(graph), -1));

    let Q = create_queue();

    Q.add_back(key);
    marks[key] = key;

    while Q.size() > 0u {
        let t = Q.pop_front();

        graph[t].each_key() {|k| 
            if marks[k] == -1 {
                marks[k] = t;
                Q.add_back(k);
            }
            true
        };
    }

    vec::from_mut(marks)
}

fn main() {
    let scale = 14u;

    let start = time::precise_time_s();
    let edges = make_edges(scale, 16u);
    let stop = time::precise_time_s();

    io::stdout().write_line(#fmt("Generated %? edges in %? seconds.",
                                 vec::len(edges), stop - start));

    let start = time::precise_time_s();
    let graph = make_graph(1u << scale, edges);
    let stop = time::precise_time_s();

    let mut total_edges = 0u;
    vec::each(graph) {|edges| total_edges += edges.size(); true };

    io::stdout().write_line(#fmt("Generated graph with %? edges in %? seconds.",
                                 total_edges / 2u,
                                 stop - start));
    
    let start = time::precise_time_s();
    let bfs_tree = bfs(graph, 0);
    let stop = time::precise_time_s();

    io::stdout().write_line(#fmt("BFS completed in %? seconds.",
                                 stop - start));
}