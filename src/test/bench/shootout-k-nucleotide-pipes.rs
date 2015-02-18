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

#![feature(box_syntax)]

use std::ascii::{AsciiExt, OwnedAsciiExt};
use std::cmp::Ordering::{self, Less, Greater, Equal};
use std::collections::HashMap;
use std::mem::replace;
use std::num::Float;
use std::option;
use std::os;
use std::env;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

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
   for (key, &val) in mm {
      pairs.push(((*key).clone(), pct(val, total)));
   }

   let pairs_sorted = sortKV(pairs);

   let mut buffer = String::new();
   for &(ref k, v) in &pairs_sorted {
       buffer.push_str(&format!("{:?} {:0.3}\n",
                                k.to_ascii_uppercase(),
                                v));
   }

   return buffer
}

// given a map, search for the frequency of a pattern
fn find(mm: &HashMap<Vec<u8> , uint>, key: String) -> uint {
   let key = key.into_ascii_lowercase();
   match mm.get(key.as_bytes()) {
      option::Option::None      => { return 0; }
      option::Option::Some(&num) => { return num; }
   }
}

// given a map, increment the counter for a key
fn update_freq(mm: &mut HashMap<Vec<u8> , uint>, key: &[u8]) {
    let key = key.to_vec();
    let newval = match mm.remove(&key) {
        Some(v) => v + 1,
        None => 1
    };
    mm.insert(key, newval);
}

// given a Vec<u8>, for each window call a function
// i.e., for "hello" and windows of size four,
// run it("hell") and it("ello"), then return "llo"
fn windows_with_carry<F>(bb: &[u8], nn: uint, mut it: F) -> Vec<u8> where
    F: FnMut(&[u8]),
{
   let mut ii = 0;

   let len = bb.len();
   while ii < len - (nn - 1) {
      it(&bb[ii..ii+nn]);
      ii += 1;
   }

   return bb[len - (nn - 1)..len].to_vec();
}

fn make_sequence_processor(sz: uint,
                           from_parent: &Receiver<Vec<u8>>,
                           to_parent: &Sender<String>) {
   let mut freqs: HashMap<Vec<u8>, uint> = HashMap::new();
   let mut carry = Vec::new();
   let mut total: uint = 0;

   let mut line: Vec<u8>;

   loop {

       line = from_parent.recv().unwrap();
       if line == Vec::new() { break; }

       carry.push_all(&line);
       carry = windows_with_carry(&carry, sz, |window| {
           update_freq(&mut freqs, window);
           total += 1;
       });
   }

   let buffer = match sz {
       1 => { sort_and_fmt(&freqs, total) }
       2 => { sort_and_fmt(&freqs, total) }
       3 => { format!("{}\t{}", find(&freqs, "GGT".to_string()), "GGT") }
       4 => { format!("{}\t{}", find(&freqs, "GGTA".to_string()), "GGTA") }
       6 => { format!("{}\t{}", find(&freqs, "GGTATT".to_string()), "GGTATT") }
      12 => { format!("{}\t{}", find(&freqs, "GGTATTTTAATT".to_string()), "GGTATTTTAATT") }
      18 => { format!("{}\t{}", find(&freqs, "GGTATTTTAATTTATAGT".to_string()),
                       "GGTATTTTAATTTATAGT") }
       _ => { "".to_string() }
   };

    to_parent.send(buffer).unwrap();
}

// given a FASTA file on stdin, process sequence THREE
fn main() {
    use std::old_io::{stdio, MemReader, BufferedReader};

    let rdr = if env::var_os("RUST_BENCH").is_some() {
        let foo = include_bytes!("shootout-k-nucleotide.data");
        box MemReader::new(foo.to_vec()) as Box<Reader>
    } else {
        box stdio::stdin() as Box<Reader>
    };
    let mut rdr = BufferedReader::new(rdr);

    // initialize each sequence sorter
    let sizes: Vec<usize> = vec!(1,2,3,4,6,12,18);
    let mut streams = (0..sizes.len()).map(|_| {
        Some(channel::<String>())
    }).collect::<Vec<_>>();
    let mut from_child = Vec::new();
    let to_child  = sizes.iter().zip(streams.iter_mut()).map(|(sz, stream_ref)| {
        let sz = *sz;
        let stream = replace(stream_ref, None);
        let (to_parent_, from_child_) = stream.unwrap();

        from_child.push(from_child_);

        let (to_child, from_parent) = channel();

        thread::spawn(move|| {
            make_sequence_processor(sz, &from_parent, &to_parent_);
        });

        to_child
    }).collect::<Vec<Sender<Vec<u8>>>>();


   // latch stores true after we've started
   // reading the sequence of interest
   let mut proc_mode = false;

   for line in rdr.lines() {
       let line = line.unwrap().trim().to_string();

       if line.len() == 0 { continue; }

       match (line.as_bytes()[0] as char, proc_mode) {

           // start processing if this is the one
           ('>', false) => {
               match line[1..].find_str("THREE") {
                   Some(_) => { proc_mode = true; }
                   None    => { }
               }
           }

           // break our processing
           ('>', true) => { break; }

           // process the sequence for k-mers
           (_, true) => {
               let line_bytes = line.as_bytes();

               for (ii, _sz) in sizes.iter().enumerate() {
                   let lb = line_bytes.to_vec();
                   to_child[ii].send(lb).unwrap();
               }
           }

           // whatever
           _ => { }
       }
   }

   // finish...
   for (ii, _sz) in sizes.iter().enumerate() {
       to_child[ii].send(Vec::new()).unwrap();
   }

   // now fetch and print result messages
   for (ii, _sz) in sizes.iter().enumerate() {
       println!("{:?}", from_child[ii].recv().unwrap());
   }
}
