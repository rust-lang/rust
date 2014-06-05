// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10393)
// ignore-pretty very bad with line comments

// multi tasking k-nucleotide

extern crate collections;

use std::collections::HashMap;
use std::mem::replace;
use std::option;
use std::os;
use std::string::String;

fn f64_cmp(x: f64, y: f64) -> Ordering {
    // arbitrarily decide that NaNs are larger than everything.
    if y.is_nan() {
        Less
    } else if x.is_nan() {
        Greater
    } else if x < y {
        Less
    } else if x == y {
        Equal
    } else {
        Greater
    }
}

// given a map, print a sorted version of it
fn sort_and_fmt(mm: &HashMap<Vec<u8> , uint>, total: uint) -> String {
   fn pct(xx: uint, yy: uint) -> f64 {
      return (xx as f64) * 100.0 / (yy as f64);
   }

   // sort by key, then by value
   fn sortKV(mut orig: Vec<(Vec<u8> ,f64)> ) -> Vec<(Vec<u8> ,f64)> {
        orig.sort_by(|&(ref a, _), &(ref b, _)| a.cmp(b));
        orig.sort_by(|&(_, a), &(_, b)| f64_cmp(b, a));
        orig
   }

   let mut pairs = Vec::new();

   // map -> [(k,%)]
   for (key, &val) in mm.iter() {
      pairs.push(((*key).clone(), pct(val, total)));
   }

   let pairs_sorted = sortKV(pairs);

   let mut buffer = String::new();
   for &(ref k, v) in pairs_sorted.iter() {
       buffer.push_str(format!("{} {:0.3f}\n",
                               k.as_slice()
                               .to_ascii()
                               .to_upper()
                               .into_str(), v).as_slice());
   }

   return buffer
}

// given a map, search for the frequency of a pattern
fn find(mm: &HashMap<Vec<u8> , uint>, key: String) -> uint {
   let key = key.to_owned().into_ascii().as_slice().to_lower().into_str();
   match mm.find_equiv(&key.as_bytes()) {
      option::None      => { return 0u; }
      option::Some(&num) => { return num; }
   }
}

// given a map, increment the counter for a key
fn update_freq(mm: &mut HashMap<Vec<u8> , uint>, key: &[u8]) {
    let key = Vec::from_slice(key);
    let newval = match mm.pop(&key) {
        Some(v) => v + 1,
        None => 1
    };
    mm.insert(key, newval);
}

// given a Vec<u8>, for each window call a function
// i.e., for "hello" and windows of size four,
// run it("hell") and it("ello"), then return "llo"
fn windows_with_carry(bb: &[u8], nn: uint, it: |window: &[u8]|) -> Vec<u8> {
   let mut ii = 0u;

   let len = bb.len();
   while ii < len - (nn - 1u) {
      it(bb.slice(ii, ii+nn));
      ii += 1u;
   }

   return Vec::from_slice(bb.slice(len - (nn - 1u), len));
}

fn make_sequence_processor(sz: uint,
                           from_parent: &Receiver<Vec<u8>>,
                           to_parent: &Sender<String>) {
   let mut freqs: HashMap<Vec<u8>, uint> = HashMap::new();
   let mut carry = Vec::new();
   let mut total: uint = 0u;

   let mut line: Vec<u8>;

   loop {

      line = from_parent.recv();
      if line == Vec::new() { break; }

       carry = windows_with_carry(carry.append(line.as_slice()).as_slice(),
                                  sz,
                                  |window| {
         update_freq(&mut freqs, window);
         total += 1u;
      });
   }

   let buffer = match sz {
       1u => { sort_and_fmt(&freqs, total) }
       2u => { sort_and_fmt(&freqs, total) }
       3u => { format!("{}\t{}", find(&freqs, "GGT".to_string()), "GGT") }
       4u => { format!("{}\t{}", find(&freqs, "GGTA".to_string()), "GGTA") }
       6u => { format!("{}\t{}", find(&freqs, "GGTATT".to_string()), "GGTATT") }
      12u => { format!("{}\t{}", find(&freqs, "GGTATTTTAATT".to_string()), "GGTATTTTAATT") }
      18u => { format!("{}\t{}", find(&freqs, "GGTATTTTAATTTATAGT".to_string()),
                       "GGTATTTTAATTTATAGT") }
        _ => { "".to_string() }
   };

    to_parent.send(buffer);
}

// given a FASTA file on stdin, process sequence THREE
fn main() {
    use std::io::{stdio, MemReader, BufferedReader};

    let rdr = if os::getenv("RUST_BENCH").is_some() {
        let foo = include_bin!("shootout-k-nucleotide.data");
        box MemReader::new(Vec::from_slice(foo)) as Box<Reader>
    } else {
        box stdio::stdin() as Box<Reader>
    };
    let mut rdr = BufferedReader::new(rdr);

    // initialize each sequence sorter
    let sizes = vec!(1u,2,3,4,6,12,18);
    let mut streams = Vec::from_fn(sizes.len(), |_| Some(channel::<String>()));
    let mut from_child = Vec::new();
    let to_child  = sizes.iter().zip(streams.mut_iter()).map(|(sz, stream_ref)| {
        let sz = *sz;
        let stream = replace(stream_ref, None);
        let (to_parent_, from_child_) = stream.unwrap();

        from_child.push(from_child_);

        let (to_child, from_parent) = channel();

        spawn(proc() {
            make_sequence_processor(sz, &from_parent, &to_parent_);
        });

        to_child
    }).collect::<Vec<Sender<Vec<u8> >> >();


   // latch stores true after we've started
   // reading the sequence of interest
   let mut proc_mode = false;

   for line in rdr.lines() {
       let line = line.unwrap().as_slice().trim().to_owned();

       if line.len() == 0u { continue; }

       match (line.as_slice()[0] as char, proc_mode) {

           // start processing if this is the one
           ('>', false) => {
               match line.as_slice().slice_from(1).find_str("THREE") {
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
                   let lb = Vec::from_slice(line_bytes);
                   to_child.get(ii).send(lb);
               }
           }

           // whatever
           _ => { }
       }
   }

   // finish...
   for (ii, _sz) in sizes.iter().enumerate() {
       to_child.get(ii).send(Vec::new());
   }

   // now fetch and print result messages
   for (ii, _sz) in sizes.iter().enumerate() {
       println!("{}", from_child.get(ii).recv());
   }
}
