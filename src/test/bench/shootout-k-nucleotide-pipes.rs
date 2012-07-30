// xfail-pretty

// multi tasking k-nucleotide

import io::reader_util;

use std;
import std::map;
import std::map::hashmap;
import std::sort;

import pipes::{stream, port, chan};

// given a map, print a sorted version of it
fn sort_and_fmt(mm: hashmap<~[u8], uint>, total: uint) -> ~str { 
   fn pct(xx: uint, yy: uint) -> float {
      ret (xx as float) * 100f / (yy as float);
   }

   fn le_by_val<TT: copy, UU: copy>(kv0: (TT,UU), kv1: (TT,UU)) -> bool {
      let (_, v0) = kv0;
      let (_, v1) = kv1;
      ret v0 >= v1;
   }

   fn le_by_key<TT: copy, UU: copy>(kv0: (TT,UU), kv1: (TT,UU)) -> bool {
      let (k0, _) = kv0;
      let (k1, _) = kv1;
      ret k0 <= k1;
   }

   // sort by key, then by value
   fn sortKV<TT: copy, UU: copy>(orig: ~[(TT,UU)]) -> ~[(TT,UU)] {
      ret sort::merge_sort(le_by_val, sort::merge_sort(le_by_key, orig));
   }

   let mut pairs = ~[];

   // map -> [(k,%)]
   mm.each(fn&(key: ~[u8], val: uint) -> bool {
      vec::push(pairs, (key, pct(val, total)));
      ret true;
   });

   let pairs_sorted = sortKV(pairs);
   
   let mut buffer = ~"";

   pairs_sorted.each(fn&(kv: (~[u8], float)) -> bool unsafe {
      let (k,v) = kv;
      buffer += (fmt!{"%s %0.3f\n", str::to_upper(str::unsafe::from_bytes(k)), v});
      ret true;
   });

   ret buffer;
}

// given a map, search for the frequency of a pattern
fn find(mm: hashmap<~[u8], uint>, key: ~str) -> uint {
   alt mm.find(str::bytes(str::to_lower(key))) {
      option::none      { ret 0u; }
      option::some(num) { ret num; }
   }
}

// given a map, increment the counter for a key
fn update_freq(mm: hashmap<~[u8], uint>, key: &[u8]) {
    let key = vec::slice(key, 0, key.len());
    alt mm.find(key) {
      option::none      { mm.insert(key, 1u      ); }
      option::some(val) { mm.insert(key, 1u + val); }
    }
}

// given a ~[u8], for each window call a function
// i.e., for "hello" and windows of size four,
// run it("hell") and it("ello"), then return "llo"
fn windows_with_carry(bb: &[u8], nn: uint,
                      it: fn(window: &[u8])) -> ~[u8] {
   let mut ii = 0u;

   let len = vec::len(bb);
   while ii < len - (nn - 1u) {
      it(vec::view(bb, ii, ii+nn));
      ii += 1u;
   }

   ret vec::slice(bb, len - (nn - 1u), len); 
}

fn make_sequence_processor(sz: uint, from_parent: pipes::port<~[u8]>,
                           to_parent: pipes::chan<~str>) {
   
   let freqs: hashmap<~[u8], uint> = map::bytes_hash();
   let mut carry: ~[u8] = ~[];
   let mut total: uint = 0u;

   let mut line: ~[u8];

   loop {

      line = from_parent.recv();
      if line == ~[] { break; }

       carry = windows_with_carry(carry + line, sz, |window| {
         update_freq(freqs, window);
         total += 1u;
      });
   }

   let buffer = alt sz { 
       1u { sort_and_fmt(freqs, total) }
       2u { sort_and_fmt(freqs, total) }
       3u { fmt!{"%u\t%s", find(freqs, ~"GGT"), ~"GGT"} }
       4u { fmt!{"%u\t%s", find(freqs, ~"GGTA"), ~"GGTA"} }
       6u { fmt!{"%u\t%s", find(freqs, ~"GGTATT"), ~"GGTATT"} }
      12u { fmt!{"%u\t%s", find(freqs, ~"GGTATTTTAATT"), ~"GGTATTTTAATT"} }
      18u { fmt!{"%u\t%s", find(freqs, ~"GGTATTTTAATTTATAGT"), ~"GGTATTTTAATTTATAGT"} }
        _ { ~"" }
   };

   //comm::send(to_parent, fmt!{"yay{%u}", sz});
    to_parent.send(buffer);
}

// given a FASTA file on stdin, process sequence THREE
fn main(args: ~[~str]) {
   let rdr = if os::getenv(~"RUST_BENCH").is_some() {
       // FIXME: Using this compile-time env variable is a crummy way to
       // get to this massive data set, but #include_bin chokes on it (#2598)
       let path = path::connect(
           env!{"CFG_SRC_DIR"},
           ~"src/test/bench/shootout-k-nucleotide.data"
           );
       result::get(io::file_reader(path))
   } else {
      io::stdin()
   };



   // initialize each sequence sorter
   let sizes = ~[1u,2u,3u,4u,6u,12u,18u];
    let streams = vec::map(sizes, |_sz| some(stream()));
    let streams = vec::to_mut(streams);
    let mut from_child = ~[];
    let to_child   = vec::mapi(sizes, |ii, sz| {
        let mut stream = none;
        stream <-> streams[ii];
        let (to_parent_, from_child_) = option::unwrap(stream);

        vec::push(from_child, from_child_);

        let (to_child, from_parent) = pipes::stream();

        do task::spawn_with(from_parent) |from_parent| {
            make_sequence_processor(sz, from_parent, to_parent_);
        };
        
        to_child
    });
         
   
   // latch stores true after we've started
   // reading the sequence of interest
   let mut proc_mode = false;

   while !rdr.eof() {
      let line: ~str = rdr.read_line();

      if str::len(line) == 0u { again; }

      alt (line[0], proc_mode) {

         // start processing if this is the one
         ('>' as u8, false) {
            alt str::find_str_from(line, ~"THREE", 1u) {
               option::some(_) { proc_mode = true; }
               option::none    { }
            }
         }

         // break our processing
         ('>' as u8, true) { break; }

         // process the sequence for k-mers
         (_, true) {
            let line_bytes = str::bytes(line);

           for sizes.eachi |ii, _sz| {
               let mut lb = line_bytes;
               to_child[ii].send(lb);
            }
         }

         // whatever
         _ { }
      }
   }

   // finish...
    for sizes.eachi |ii, _sz| {
      to_child[ii].send(~[]);
   }

   // now fetch and print result messages
    for sizes.eachi |ii, _sz| {
      io::println(from_child[ii].recv());
   }
}

