// xfail-pretty

// multi tasking k-nucleotide

#[legacy_modes];

extern mod std;
use std::map;
use std::map::HashMap;
use std::sort;
use io::ReaderUtil;
use pipes::{stream, Port, Chan};
use cmp::Ord;

// given a map, print a sorted version of it
fn sort_and_fmt(mm: HashMap<~[u8], uint>, total: uint) -> ~str { 
   fn pct(xx: uint, yy: uint) -> float {
      return (xx as float) * 100f / (yy as float);
   }

   pure fn le_by_val<TT: Copy, UU: Copy Ord>(kv0: &(TT,UU),
                                         kv1: &(TT,UU)) -> bool {
      let (_, v0) = *kv0;
      let (_, v1) = *kv1;
      return v0 >= v1;
   }

   pure fn le_by_key<TT: Copy Ord, UU: Copy>(kv0: &(TT,UU),
                                         kv1: &(TT,UU)) -> bool {
      let (k0, _) = *kv0;
      let (k1, _) = *kv1;
      return k0 <= k1;
   }

   // sort by key, then by value
   fn sortKV<TT: Copy Ord, UU: Copy Ord>(orig: ~[(TT,UU)]) -> ~[(TT,UU)] {
      return sort::merge_sort(sort::merge_sort(orig, le_by_key), le_by_val);
   }

   let mut pairs = ~[];

   // map -> [(k,%)]
   mm.each(fn&(key: ~[u8], val: uint) -> bool {
      pairs.push((key, pct(val, total)));
      return true;
   });

   let pairs_sorted = sortKV(pairs);

   let mut buffer = ~"";

   for pairs_sorted.each |kv| {
       let (k,v) = *kv;
       unsafe {
           buffer += (fmt!("%s %0.3f\n", str::to_upper(str::raw::from_bytes(k)), v));
       }
   }

   return buffer;
}

// given a map, search for the frequency of a pattern
fn find(mm: HashMap<~[u8], uint>, key: ~str) -> uint {
   match mm.find(str::to_bytes(str::to_lower(key))) {
      option::None      => { return 0u; }
      option::Some(num) => { return num; }
   }
}

// given a map, increment the counter for a key
fn update_freq(mm: HashMap<~[u8], uint>, key: &[u8]) {
    let key = vec::slice(key, 0, key.len());
    match mm.find(key) {
      option::None      => { mm.insert(key, 1u      ); }
      option::Some(val) => { mm.insert(key, 1u + val); }
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

   return vec::slice(bb, len - (nn - 1u), len); 
}

fn make_sequence_processor(sz: uint, from_parent: pipes::Port<~[u8]>,
                           to_parent: pipes::Chan<~str>) {
   
   let freqs: HashMap<~[u8], uint> = map::HashMap();
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

   let buffer = match sz { 
       1u => { sort_and_fmt(freqs, total) }
       2u => { sort_and_fmt(freqs, total) }
       3u => { fmt!("%u\t%s", find(freqs, ~"GGT"), ~"GGT") }
       4u => { fmt!("%u\t%s", find(freqs, ~"GGTA"), ~"GGTA") }
       6u => { fmt!("%u\t%s", find(freqs, ~"GGTATT"), ~"GGTATT") }
      12u => { fmt!("%u\t%s", find(freqs, ~"GGTATTTTAATT"), ~"GGTATTTTAATT") }
      18u => { fmt!("%u\t%s", find(freqs, ~"GGTATTTTAATTTATAGT"), ~"GGTATTTTAATTTATAGT") }
        _ => { ~"" }
   };

   //comm::send(to_parent, fmt!("yay{%u}", sz));
    to_parent.send(move buffer);
}

// given a FASTA file on stdin, process sequence THREE
fn main() {
    let args = os::args();
   let rdr = if os::getenv(~"RUST_BENCH").is_some() {
       // FIXME: Using this compile-time env variable is a crummy way to
       // get to this massive data set, but include_bin! chokes on it (#2598)
       let path = Path(env!("CFG_SRC_DIR"))
           .push_rel(&Path("src/test/bench/shootout-k-nucleotide.data"));
       result::get(&io::file_reader(&path))
   } else {
      io::stdin()
   };



   // initialize each sequence sorter
   let sizes = ~[1,2,3,4,6,12,18];
    let streams = vec::map(sizes, |_sz| Some(stream()));
    let streams = vec::to_mut(move streams);
    let mut from_child = ~[];
    let to_child   = vec::mapi(sizes, |ii, sz| {
        let sz = *sz;
        let mut stream = None;
        stream <-> streams[ii];
        let (to_parent_, from_child_) = option::unwrap(move stream);

        from_child.push(move from_child_);

        let (to_child, from_parent) = pipes::stream();

        do task::spawn_with(move from_parent) |move to_parent_, from_parent| {
            make_sequence_processor(sz, from_parent, to_parent_);
        };
        
        move to_child
    });
         
   
   // latch stores true after we've started
   // reading the sequence of interest
   let mut proc_mode = false;

   while !rdr.eof() {
      let line: ~str = rdr.read_line();

      if str::len(line) == 0u { loop; }

      match (line[0], proc_mode) {

         // start processing if this is the one
         ('>' as u8, false) => {
            match str::find_str_from(line, ~"THREE", 1u) {
               option::Some(_) => { proc_mode = true; }
               option::None    => { }
            }
         }

         // break our processing
         ('>' as u8, true) => { break; }

         // process the sequence for k-mers
         (_, true) => {
            let line_bytes = str::to_bytes(line);

           for sizes.eachi |ii, _sz| {
               let mut lb = line_bytes;
               to_child[ii].send(lb);
            }
         }

         // whatever
         _ => { }
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

