/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

use std;

import std::io;
import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;
import std::map;

fn map(str filename, map_reduce::putter emit) {
    emit(filename, "1");
}

mod map_reduce {
    export putter;
    export mapper;
    export map_reduce;

    type putter = fn(str, str) -> ();

    type mapper = fn(str, putter);

    tag ctrl_proto {
        find_reducer(str, chan[int]);
        mapper_done;
    }

    fn start_mappers(chan[ctrl_proto] ctrl,
                     vec[str] inputs) {
        for(str i in inputs) {
            spawn map_task(ctrl, i);
        }
    }

    fn map_task(chan[ctrl_proto] ctrl,
                str input) {

        auto intermediates = map::new_str_hash();

        fn emit(&map::hashmap[str, int] im,
                chan[ctrl_proto] ctrl,
                str key, str val) {
            auto c;
            alt(im.find(key)) {
                case(some(?_c)) {
                    c = _c
                }
                case(none) {
                    auto p = port();
                    log_err "sending find_reducer";
                    ctrl <| find_reducer(key, chan(p));
                    log_err "receiving";
                    p |> c;
                    log_err c;
                    im.insert(key, c);
                }
            }
        }

        map(input, bind emit(intermediates, ctrl, _, _));
        ctrl <| mapper_done;
    }

    fn map_reduce (vec[str] inputs) {
        auto ctrl = port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let map::hashmap[str, int] reducers;

        reducers = map::new_str_hash();

        start_mappers(chan(ctrl), inputs);

        auto num_mappers = vec::len(inputs) as int;

        while(num_mappers > 0) {
            auto m;
            ctrl |> m;

            alt(m) {
                case(mapper_done) { num_mappers -= 1; }
                case(find_reducer(?k, ?cc)) {
                    auto c;
                    alt(reducers.find(k)) {
                        case(some(?_c)) { c = _c; }
                        case(none) {
                            c = 0;
                        }
                    }
                    cc <| c;
                }
            }
        }
    }
}

fn main() {
    map_reduce::map_reduce(["../src/test/run-pass/hashmap-memory.rs"]);
}
