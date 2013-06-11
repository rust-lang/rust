// xfail-test

extern mod extra;

use std::cast::transmute;
use std::i32::range;
use std::libc::{STDIN_FILENO, c_int, fdopen, fgets, fileno, fopen, fstat};
use std::libc::{stat, strlen};
use std::ptr::null;
use std::unstable::intrinsics::init;
use std::vec::{reverse, slice};
use extra::sort::quick_sort3;

static LINE_LEN: uint = 80;
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

#[deriving(Eq, Ord)]
struct Code(u64);

impl Code {
    fn hash(&self) -> u64 {
        **self
    }

    #[inline(always)]
    fn push_char(&self, c: u8) -> Code {
        Code((**self << 2) + (pack_symbol(c) as u64))
    }

    fn rotate(&self, c: u8, frame: i32) -> Code {
        Code(*self.push_char(c) & ((1u64 << (2 * (frame as u64))) - 1))
    }

    fn pack(string: &str) -> Code {
        let mut code = Code(0u64);
        for uint::range(0, string.len()) |i| {
            code = code.push_char(string[i]);
        }
        code
    }

    // XXX: Inefficient.
    fn unpack(&self, frame: i32) -> ~str {
        let mut key = **self;
        let mut result = ~[];
        for (frame as uint).times {
            result.push(unpack_symbol((key as u8) & 3));
            key >>= 2;
        }

        reverse(result);
        str::from_bytes(result)
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
        println(fmt!("%d\t%s", entry.count as int, **self));
    }
}

struct Entry {
    code: Code,
    count: i32,
    next: Option<~Entry>,
}

struct Table {
    count: i32,
    items: [Option<~Entry>, ..TABLE_SIZE]
}

impl Table {
    fn new() -> Table {
        Table {
            count: 0,
            items: [ None, ..TABLE_SIZE ],
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
        let index = *key % (TABLE_SIZE as u64);

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
            let mut entry = &mut *self.items[index].get_mut_ref();
            if entry.code == key {
                c.f(*entry);
                return;
            }

            Table::search_remainder(*entry, key, c)
        }
    }

    fn each(&self, f: &fn(entry: &Entry) -> bool) {
        for self.items.each |item| {
            match *item {
                None => {}
                Some(ref item) => {
                    let mut item: &Entry = *item;
                    loop {
                        if !f(item) {
                            return;
                        }

                        match item.next {
                            None => break,
                            Some(ref next_item) => item = &**next_item,
                        }
                    }
                }
            };
        }
    }
}

// Main program

fn pack_symbol(c: u8) -> u8 {
    match c {
        'a' as u8 | 'A' as u8 => 0,
        'c' as u8 | 'C' as u8 => 1,
        'g' as u8 | 'G' as u8 => 2,
        't' as u8 | 'T' as u8 => 3,
        _ => fail!(c.to_str())
    }
}

fn unpack_symbol(c: u8) -> u8 {
    TABLE[c]
}

fn next_char<'a>(mut buf: &'a [u8]) -> &'a [u8] {
    loop {
        buf = slice(buf, 1, buf.len());
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

#[inline(never)]
fn read_stdin() -> ~[u8] {
    unsafe {
        let mode = "r";
        //let stdin = fdopen(STDIN_FILENO as c_int, transmute(&mode[0]));
        let path = "knucleotide-input.txt";
        let stdin = fopen(transmute(&path[0]), transmute(&mode[0]));

        let mut st: stat = init();
        fstat(fileno(stdin), &mut st);
        let mut buf = vec::from_elem(st.st_size as uint, 0);

        let header = ">THREE".as_bytes();

        {
            let mut window: &mut [u8] = buf;
            loop {
                fgets(transmute(&mut window[0]), LINE_LEN as c_int, stdin);

                {
                    if vec::slice(window, 0, 6) == header {
                        break;
                    }
                }
            }

            while fgets(transmute(&mut window[0]),
                        LINE_LEN as c_int,
                        stdin) != null() {
                window = vec::mut_slice(window,
                                        strlen(transmute(&window[0])) as uint,
                                        window.len());
            }
        }

        buf
    }
}

#[inline(never)]
#[fixed_stack_segment]
fn generate_frequencies(frequencies: &mut Table,
                        mut input: &[u8],
                        frame: i32) {
    let mut code = Code(0);

    // Pull first frame.
    for (frame as uint).times {
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

#[inline(never)]
#[fixed_stack_segment]
fn print_frequencies(frequencies: &Table, frame: i32) {
    let mut vector = ~[];
    for frequencies.each |entry| {
        vector.push((entry.code, entry.count));
    }
    quick_sort3(vector);

    let mut total_count = 0;
    for vector.each |&(_, count)| {
        total_count += count;
    }

    for vector.each |&(key, count)| {
        println(fmt!("%s %.3f",
                     key.unpack(frame),
                     (count as float * 100.0) / (total_count as float)));
    }
}

fn print_occurrences(frequencies: &mut Table, occurrence: &'static str) {
    frequencies.lookup(Code::pack(occurrence), PrintCallback(occurrence))
}

#[fixed_stack_segment]
fn main() {
    let input = read_stdin();

    let mut frequencies = ~Table::new();
    generate_frequencies(frequencies, input, 1);
    print_frequencies(frequencies, 1);

    *frequencies = Table::new();
    generate_frequencies(frequencies, input, 2);
    print_frequencies(frequencies, 2);

    for range(0, 5) |i| {
        let occurrence = OCCURRENCES[i];
        *frequencies = Table::new();
        generate_frequencies(frequencies,
                             input,
                             occurrence.len() as i32);
        print_occurrences(frequencies, occurrence);
    }
}
