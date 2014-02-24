// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

use std::str;
use std::vec;

static TABLE: [u8, ..4] = [ 'A' as u8, 'C' as u8, 'G' as u8, 'T' as u8 ];
static TABLE_SIZE: uint = 2 << 16;

static OCCURRENCES: [&'static str, ..5] = [
    "GGT",
    "GGTA",
    "GGTATT",
    "GGTATTTTAATT",
    "GGTATTTTAATTTATAGT",
];

// Code implementation

#[deriving(Eq, Ord, TotalOrd, TotalEq)]
struct Code(u64);

impl Code {
    fn hash(&self) -> u64 {
        let Code(ret) = *self;
        return ret;
    }

    fn push_char(&self, c: u8) -> Code {
        Code((self.hash() << 2) + (pack_symbol(c) as u64))
    }

    fn rotate(&self, c: u8, frame: i32) -> Code {
        Code(self.push_char(c).hash() & ((1u64 << (2 * (frame as u64))) - 1))
    }

    fn pack(string: &str) -> Code {
        string.bytes().fold(Code(0u64), |a, b| a.push_char(b))
    }

    // FIXME: Inefficient.
    fn unpack(&self, frame: i32) -> ~str {
        let mut key = self.hash();
        let mut result = ~[];
        for _ in range(0, frame) {
            result.push(unpack_symbol((key as u8) & 3));
            key >>= 2;
        }

        result.reverse();
        str::from_utf8_owned(result).unwrap()
    }
}

// Hash table implementation

trait TableCallback {
    fn f(&self, entry: &mut Entry);
}

struct BumpCallback;

impl TableCallback for BumpCallback {
    fn f(&self, entry: &mut Entry) {
        entry.count += 1;
    }
}

struct PrintCallback(&'static str);

impl TableCallback for PrintCallback {
    fn f(&self, entry: &mut Entry) {
        let PrintCallback(s) = *self;
        println!("{}\t{}", entry.count as int, s);
    }
}

struct Entry {
    code: Code,
    count: i32,
    next: Option<~Entry>,
}

struct Table {
    count: i32,
    items: ~[Option<~Entry>]
}

struct Items<'a> {
    cur: Option<&'a Entry>,
    items: vec::Items<'a, Option<~Entry>>,
}

impl Table {
    fn new() -> Table {
        Table {
            count: 0,
            items: vec::from_fn(TABLE_SIZE, |_| None),
        }
    }

    fn search_remainder<C:TableCallback>(item: &mut Entry, key: Code, c: C) {
        match item.next {
            None => {
                let mut entry = ~Entry {
                    code: key,
                    count: 0,
                    next: None,
                };
                c.f(entry);
                item.next = Some(entry);
            }
            Some(ref mut entry) => {
                if entry.code == key {
                    c.f(*entry);
                    return;
                }

                Table::search_remainder(*entry, key, c)
            }
        }
    }

    fn lookup<C:TableCallback>(&mut self, key: Code, c: C) {
        let index = key.hash() % (TABLE_SIZE as u64);

        {
            if self.items[index].is_none() {
                let mut entry = ~Entry {
                    code: key,
                    count: 0,
                    next: None,
                };
                c.f(entry);
                self.items[index] = Some(entry);
                return;
            }
        }

        {
            let entry = &mut *self.items[index].get_mut_ref();
            if entry.code == key {
                c.f(*entry);
                return;
            }

            Table::search_remainder(*entry, key, c)
        }
    }

    fn iter<'a>(&'a self) -> Items<'a> {
        Items { cur: None, items: self.items.iter() }
    }
}

impl<'a> Iterator<&'a Entry> for Items<'a> {
    fn next(&mut self) -> Option<&'a Entry> {
        let ret = match self.cur {
            None => {
                let i;
                loop {
                    match self.items.next() {
                        None => return None,
                        Some(&None) => {}
                        Some(&Some(ref a)) => { i = &**a; break }
                    }
                }
                self.cur = Some(&*i);
                &*i
            }
            Some(c) => c
        };
        match ret.next {
            None => { self.cur = None; }
            Some(ref next) => { self.cur = Some(&**next); }
        }
        return Some(ret);
    }
}

// Main program

fn pack_symbol(c: u8) -> u8 {
    match c as char {
        'a' | 'A' => 0,
        'c' | 'C' => 1,
        'g' | 'G' => 2,
        't' | 'T' => 3,
        _ => fail!("{}", c as char),
    }
}

fn unpack_symbol(c: u8) -> u8 {
    TABLE[c]
}

fn next_char<'a>(mut buf: &'a [u8]) -> &'a [u8] {
    loop {
        buf = buf.slice(1, buf.len());
        if buf.len() == 0 {
            break;
        }
        if buf[0] != (' ' as u8) && buf[0] != ('\t' as u8) &&
                buf[0] != ('\n' as u8) && buf[0] != 0 {
            break;
        }
    }
    buf
}

fn generate_frequencies(frequencies: &mut Table,
                        mut input: &[u8],
                        frame: i32) {
    let mut code = Code(0);

    // Pull first frame.
    for _ in range(0, frame) {
        code = code.push_char(input[0]);
        input = next_char(input);
    }
    frequencies.lookup(code, BumpCallback);

    while input.len() != 0 && input[0] != ('>' as u8) {
        code = code.rotate(input[0], frame);
        frequencies.lookup(code, BumpCallback);
        input = next_char(input);
    }
}

fn print_frequencies(frequencies: &Table, frame: i32) {
    let mut vector = ~[];
    for entry in frequencies.iter() {
        vector.push((entry.code, entry.count));
    }
    vector.sort();

    let mut total_count = 0;
    for &(_, count) in vector.iter() {
        total_count += count;
    }

    for &(key, count) in vector.iter() {
        println!("{} {:.3f}",
                 key.unpack(frame),
                 (count as f32 * 100.0) / (total_count as f32));
    }
}

fn print_occurrences(frequencies: &mut Table, occurrence: &'static str) {
    frequencies.lookup(Code::pack(occurrence), PrintCallback(occurrence))
}

fn main() {
    let input = include_str!("shootout-k-nucleotide.data");
    let pos = input.find_str(">THREE").unwrap();
    let pos2 = pos + input.slice_from(pos).find_str("\n").unwrap();
    let input = input.slice_from(pos2 + 1).as_bytes();

    let mut frequencies = Table::new();
    generate_frequencies(&mut frequencies, input, 1);
    print_frequencies(&frequencies, 1);

    frequencies = Table::new();
    generate_frequencies(&mut frequencies, input, 2);
    print_frequencies(&frequencies, 2);

    for occurrence in OCCURRENCES.iter() {
        frequencies = Table::new();
        generate_frequencies(&mut frequencies,
                             input,
                             occurrence.len() as i32);
        print_occurrences(&mut frequencies, *occurrence);
    }
}
