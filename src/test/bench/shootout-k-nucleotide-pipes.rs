// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10393)

// xfail-pretty the `let to_child` line gets an extra newline
// multi tasking k-nucleotide

extern mod extra;

use extra::sort;
use std::cmp::Ord;
use std::comm::{stream, Port, Chan};
use std::comm;
use std::hashmap::HashMap;
use std::option;
use std::os;
use std::io;
use std::str;
use std::task;
use std::util;
use std::vec;

// given a map, print a sorted version of it
fn sort_and_fmt(mm: &HashMap<~[u8], uint>, total: uint) -> ~str {
   fn pct(xx: uint, yy: uint) -> f64 {
      return (xx as f64) * 100.0 / (yy as f64);
   }

   fn le_by_val<TT:Clone,
                UU:Clone + Ord>(
                kv0: &(TT,UU),
                kv1: &(TT,UU))
                -> bool {
      let (_, v0) = (*kv0).clone();
      let (_, v1) = (*kv1).clone();
      return v0 >= v1;
   }

   fn le_by_key<TT:Clone + Ord,
                UU:Clone>(
                kv0: &(TT,UU),
                kv1: &(TT,UU))
                -> bool {
      let (k0, _) = (*kv0).clone();
      let (k1, _) = (*kv1).clone();
      return k0 <= k1;
   }

   // sort by key, then by value
   fn sortKV<TT:Clone + Ord, UU:Clone + Ord>(orig: ~[(TT,UU)]) -> ~[(TT,UU)] {
      return sort::merge_sort(sort::merge_sort(orig, le_by_key), le_by_val);
   }

   let mut pairs = ~[];

   // map -> [(k,%)]
   for (key, &val) in mm.iter() {
      pairs.push(((*key).clone(), pct(val, total)));
   }

   let pairs_sorted = sortKV(pairs);

   let mut buffer = ~"";

   for kv in pairs_sorted.iter() {
       let (k,v) = (*kv).clone();
       unsafe {
           let b = str::raw::from_utf8(k);
           buffer.push_str(format!("{} {:0.3f}\n",
                                   b.into_ascii().to_upper().into_str(), v));
       }
   }

   return buffer;
}

// given a map, search for the frequency of a pattern
fn find(mm: &HashMap<~[u8], uint>, key: ~str) -> uint {
   let key = key.into_ascii().to_lower().into_str();
   match mm.find_equiv(&key.as_bytes()) {
      option::None      => { return 0u; }
      option::Some(&num) => { return num; }
   }
}

// given a map, increment the counter for a key
fn update_freq(mm: &mut HashMap<~[u8], uint>, key: &[u8]) {
    let key = key.to_owned();
    let newval = match mm.pop(&key) {
        Some(v) => v + 1,
        None => 1
    };
    mm.insert(key, newval);
}

// given a ~[u8], for each window call a function
// i.e., for "hello" and windows of size four,
// run it("hell") and it("ello"), then return "llo"
fn windows_with_carry(bb: &[u8], nn: uint,
                      it: &fn(window: &[u8])) -> ~[u8] {
   let mut ii = 0u;

   let len = bb.len();
   while ii < len - (nn - 1u) {
      it(bb.slice(ii, ii+nn));
      ii += 1u;
   }

   return bb.slice(len - (nn - 1u), len).to_owned();
}

fn make_sequence_processor(sz: uint,
                           from_parent: &Port<~[u8]>,
                           to_parent: &Chan<~str>) {
   let mut freqs: HashMap<~[u8], uint> = HashMap::new();
   let mut carry: ~[u8] = ~[];
   let mut total: uint = 0u;

   let mut line: ~[u8];

   loop {

      line = from_parent.recv();
      if line == ~[] { break; }

       carry = windows_with_carry(carry + line, sz, |window| {
         update_freq(&mut freqs, window);
         total += 1u;
      });
   }

   let buffer = match sz {
       1u => { sort_and_fmt(&freqs, total) }
       2u => { sort_and_fmt(&freqs, total) }
       3u => { format!("{}\t{}", find(&freqs, ~"GGT"), "GGT") }
       4u => { format!("{}\t{}", find(&freqs, ~"GGTA"), "GGTA") }
       6u => { format!("{}\t{}", find(&freqs, ~"GGTATT"), "GGTATT") }
      12u => { format!("{}\t{}", find(&freqs, ~"GGTATTTTAATT"), "GGTATTTTAATT") }
      18u => { format!("{}\t{}", find(&freqs, ~"GGTATTTTAATTTATAGT"), "GGTATTTTAATTTATAGT") }
        _ => { ~"" }
   };

    to_parent.send(buffer);
}

// given a FASTA file on stdin, process sequence THREE
fn main() {
    use std::io::Reader;
    use std::io::native::stdio;
    use std::io::mem::MemReader;
    use std::io::buffered::BufferedReader;

    let rdr = if os::getenv("RUST_BENCH").is_some() {
        let foo = include_bin!("shootout-k-nucleotide.data");
        ~MemReader::new(foo.to_owned()) as ~Reader
    } else {
        ~stdio::stdin() as ~Reader
    };
    let mut rdr = BufferedReader::new(rdr);

    // initialize each sequence sorter
    let sizes = ~[1u,2,3,4,6,12,18];
    let mut streams = vec::from_fn(sizes.len(), |_| Some(stream::<~str>()));
    let mut from_child = ~[];
    let to_child   = do sizes.iter().zip(streams.mut_iter()).map |(sz, stream_ref)| {
        let sz = *sz;
        let stream = util::replace(stream_ref, None);
        let (from_child_, to_parent_) = stream.unwrap();

        from_child.push(from_child_);

        let (from_parent, to_child) = comm::stream();

        do task::spawn_with(from_parent) |from_parent| {
            make_sequence_processor(sz, &from_parent, &to_parent_);
        };

        to_child
    }.collect::<~[Chan<~[u8]>]>();


   // latch stores true after we've started
   // reading the sequence of interest
   let mut proc_mode = false;

   loop {
       let line = match io::ignore_io_error(|| rdr.read_line()) {
           Some(ln) => ln, None => break,
       };
       let line = line.trim().to_owned();

       if line.len() == 0u { continue; }

       match (line[0] as char, proc_mode) {

           // start processing if this is the one
           ('>', false) => {
               match line.slice_from(1).find_str("THREE") {
                   option::Some(_) => { proc_mode = true; }
                   option::None    => { }
               }
           }

           // break our processing
           ('>', true) => { break; }

           // process the sequence for k-mers
           (_, true) => {
               let line_bytes = line.as_bytes();

               for (ii, _sz) in sizes.iter().enumerate() {
                   let lb = line_bytes.to_owned();
                   to_child[ii].send(lb);
               }
           }

           // whatever
           _ => { }
       }
   }

   // finish...
   for (ii, _sz) in sizes.iter().enumerate() {
       to_child[ii].send(~[]);
   }

   // now fetch and print result messages
   for (ii, _sz) in sizes.iter().enumerate() {
       println(from_child[ii].recv());
   }
}
