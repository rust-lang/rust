/**
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

use std;

import std::io;
import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;
import std::map;

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter = fn(str, str) -> ();

    type mapper = fn(str, putter);

    type getter = fn() -> option[str];
    
    type reducer = fn(str, getter);

    
    fn map_reduce (vec[str] inputs,
                   mapper f,
                   reducer reduce) {
        auto intermediates = map::new_str_hash[vec[str]]();

        fn emit(&map::hashmap[str, vec[str]] im,
                str key, str val) {
            auto old = [];
            alt(im.remove(key)) {
                case (some(?v)) {
                    old = v;
                }
                case (none) { }
            }
            
            im.insert(key, old + [val]);
        }

        for (str i in inputs) {
            f(i, bind emit(intermediates, _, _));
        }

        fn get(vec[str] vals, &mutable uint i) -> option[str] {
            i += 1u;
            if(i <= vec::len(vals)) {
                some(vals.(i - 1u))
            }
            else {
                none
            }
        }

        for each (@tup(str, vec[str]) kv in intermediates.items()) {
            auto i = 0u;
            reduce(kv._0, bind get(kv._1, i));
        }
    }
}

fn main(vec[str] argv) {
    if(vec::len(argv) < 2u) {
        auto out = io::stdout();

        out.write_line(#fmt("Usage: %s <filename> ...", argv.(0)));
        fail;
    }

    fn map(str filename, map_reduce::putter emit) {
        auto f = io::file_reader(filename);

        while(true) {
            alt(read_word(f)) {
                case (some(?w)) { 
                    emit(w, "1");
                }
                case (none) {
                    break;
                }
            }
        }
    }

    fn reduce(str word, map_reduce::getter get) {
        auto count = 0;
        
        while(true) {
            alt(get()) {
                case(some(_)) { count += 1 }
                case(none) { break }
            }
        }

        auto out = io::stdout();
        out.write_line(#fmt("%s: %d", word, count));
    }

    map_reduce::map_reduce(vec::slice(argv, 1u, vec::len(argv)), map, reduce);
}

fn read_word(io::reader r) -> option[str] {
    auto w = "";

    while(!r.eof()) {
        auto c = r.read_char();

        if(is_word_char(c)) {
            w += str::from_char(c);
        }
        else {
            if(w != "") {
                ret some(w);
            }
        }
    }
    ret none;
}

fn is_digit(char c) -> bool {
    alt(c) {
        case ('0') { true }
        case ('1') { true }
        case ('2') { true }
        case ('3') { true }
        case ('4') { true }
        case ('5') { true }
        case ('6') { true }
        case ('7') { true }
        case ('8') { true }
        case ('9') { true }
        case (_) { false }
    }
}

fn is_alpha_lower (char c) -> bool {
    alt(c) {
        case ('a') { true }
        case ('b') { true }
        case ('c') { true }
        case ('d') { true }
        case ('e') { true }
        case ('f') { true }
        case ('g') { true }
        case ('h') { true }
        case ('i') { true }
        case ('j') { true }
        case ('k') { true }
        case ('l') { true }
        case ('m') { true }
        case ('n') { true }
        case ('o') { true }
        case ('p') { true }
        case ('q') { true }
        case ('r') { true }
        case ('s') { true }
        case ('t') { true }
        case ('u') { true }
        case ('v') { true }
        case ('w') { true }
        case ('x') { true }
        case ('y') { true }
        case ('z') { true }
        case (_) { false }
    }
}

fn is_alpha_upper (char c) -> bool {
    alt(c) {
        case ('A') { true }
        case ('B') { true }
        case ('C') { true }
        case ('D') { true }
        case ('E') { true }
        case ('F') { true }
        case ('G') { true }
        case ('H') { true }
        case ('I') { true }
        case ('J') { true }
        case ('K') { true }
        case ('L') { true }
        case ('M') { true }
        case ('N') { true }
        case ('O') { true }
        case ('P') { true }
        case ('Q') { true }
        case ('R') { true }
        case ('S') { true }
        case ('T') { true }
        case ('U') { true }
        case ('V') { true }
        case ('W') { true }
        case ('X') { true }
        case ('Y') { true }
        case ('Z') { true }
        case (_) { false }
    }
}

fn is_alpha(char c) -> bool {
    is_alpha_upper(c) || is_alpha_lower(c)
}

fn is_word_char(char c) -> bool {
    is_alpha(c) || is_digit(c) || c == '_'
}