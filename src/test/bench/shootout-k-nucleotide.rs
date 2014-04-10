// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android see #10393 #13206
// ignore-pretty

use std::strbuf::StrBuf;
use std::slice;

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

    fn rotate(&self, c: u8, frame: uint) -> Code {
        Code(self.push_char(c).hash() & ((1u64 << (2 * frame)) - 1))
    }

    fn pack(string: &str) -> Code {
        string.bytes().fold(Code(0u64), |a, b| a.push_char(b))
    }

    fn unpack(&self, frame: uint) -> StrBuf {
        let mut key = self.hash();
        let mut result = Vec::new();
        for _ in range(0, frame) {
            result.push(unpack_symbol((key as u8) & 3));
            key >>= 2;
        }

        result.reverse();
        StrBuf::from_utf8(result).unwrap()
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
    count: uint,
    next: Option<~Entry>,
}

struct Table {
    count: uint,
    items: Vec<Option<~Entry>> }

struct Items<'a> {
    cur: Option<&'a Entry>,
    items: slice::Items<'a, Option<~Entry>>,
}

impl Table {
    fn new() -> Table {
        Table {
            count: 0,
            items: Vec::from_fn(TABLE_SIZE, |_| None),
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
            if self.items.get(index as uint).is_none() {
                let mut entry = ~Entry {
                    code: key,
                    count: 0,
                    next: None,
                };
                c.f(entry);
                *self.items.get_mut(index as uint) = Some(entry);
                return;
            }
        }

        {
            let entry = &mut *self.items.get_mut(index as uint).get_mut_ref();
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
        'A' => 0,
        'C' => 1,
        'G' => 2,
        'T' => 3,
        _ => fail!("{}", c as char),
    }
}

fn unpack_symbol(c: u8) -> u8 {
    TABLE[c as uint]
}

fn generate_frequencies(frequencies: &mut Table,
                        mut input: &[u8],
                        frame: uint) {
    if input.len() < frame { return; }
    let mut code = Code(0);

    // Pull first frame.
    for _ in range(0, frame) {
        code = code.push_char(input[0]);
        input = input.slice_from(1);
    }
    frequencies.lookup(code, BumpCallback);

    while input.len() != 0 && input[0] != ('>' as u8) {
        code = code.rotate(input[0], frame);
        frequencies.lookup(code, BumpCallback);
        input = input.slice_from(1);
    }
}

fn print_frequencies(frequencies: &Table, frame: uint) {
    let mut vector = Vec::new();
    for entry in frequencies.iter() {
        vector.push((entry.count, entry.code));
    }
    vector.as_mut_slice().sort();

    let mut total_count = 0;
    for &(count, _) in vector.iter() {
        total_count += count;
    }

    for &(count, key) in vector.iter().rev() {
        println!("{} {:.3f}",
                 key.unpack(frame).as_slice(),
                 (count as f32 * 100.0) / (total_count as f32));
    }
    println!("");
}

fn print_occurrences(frequencies: &mut Table, occurrence: &'static str) {
    frequencies.lookup(Code::pack(occurrence), PrintCallback(occurrence))
}

fn get_sequence<R: Buffer>(r: &mut R, key: &str) -> Vec<u8> {
    let mut res = Vec::new();
    for l in r.lines().map(|l| l.ok().unwrap())
        .skip_while(|l| key != l.slice_to(key.len())).skip(1)
    {
        res.push_all(l.trim().as_bytes());
    }
    for b in res.mut_iter() {
        *b = b.to_ascii().to_upper().to_byte();
    }
    res
}

fn main() {
    let input = if std::os::getenv("RUST_BENCH").is_some() {
        let fd = std::io::File::open(&Path::new("shootout-k-nucleotide.data"));
        get_sequence(&mut std::io::BufferedReader::new(fd), ">THREE")
    } else {
        get_sequence(&mut std::io::stdin(), ">THREE")
    };

    let mut frequencies = Table::new();
    generate_frequencies(&mut frequencies, input.as_slice(), 1);
    print_frequencies(&frequencies, 1);

    frequencies = Table::new();
    generate_frequencies(&mut frequencies, input.as_slice(), 2);
    print_frequencies(&frequencies, 2);

    for occurrence in OCCURRENCES.iter() {
        frequencies = Table::new();
        generate_frequencies(&mut frequencies,
                             input.as_slice(),
                             occurrence.len());
        print_occurrences(&mut frequencies, *occurrence);
    }
}
