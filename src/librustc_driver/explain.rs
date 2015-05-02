// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate term;

fn parse_input(input: &str) {
    let lines : Vec<&str> = input.split('\n').collect();
    //let mut out = String::new();
    let mut total = lines.len();
    let mut t = term::stdout().unwrap();

    for line in lines {
        if line.starts_with("#") {
            t.attr(term::attr::Bold).unwrap();
            t.fg(term::color::WHITE).unwrap();
            (writeln!(t, "{}", line)).unwrap();
            t.reset().unwrap();
            continue;
        }
        let words : Vec<&str> = line.split(' ').collect();
        let mut it = 0;

        while it < words.len() {
            let word : &str = words[it];

            match word {
                "pub" | "const" | "static" | "crate" | "extern" => {
                    t.attr(term::attr::Bold).unwrap();
                    t.fg(term::color::RED).unwrap();
                    (write!(t, "{}", word)).unwrap();
                    t.reset().unwrap();
                }
                "fn" | "struct" | "mod" | "type" | "enum" | "let" | "match" | "trait" => {
                    t.attr(term::attr::Bold).unwrap();
                    t.fg(term::color::RED).unwrap();
                    (write!(t, "{} ", word)).unwrap();
                    it += 1;
                    t.fg(term::color::BLUE).unwrap();
                    if it < words.len() {
                        (write!(t, "{}", words[it])).unwrap();
                    }
                    t.reset().unwrap();
                }
                _ => {
                    if word.find(' ').is_some() {
                        let funcs : Vec<&str> = word.split('.').collect();

                        match funcs[funcs.len() - 1].find('(') {
                            Some(_) => {
                                let mut i = 0;

                                if funcs.len() > 1 {
                                    while i < funcs.len() - 2 {
                                        t.attr(term::attr::Bold).unwrap();
                                        t.fg(term::color::BLUE).unwrap();
                                        (write!(t, "{}.", funcs[i])).unwrap();
                                        t.reset().unwrap();
                                        i += 1;
                                    }
                                    if i < funcs.len() {
                                        let func_name : Vec<&str> = funcs[i].split('(').collect();
                                        t.attr(term::attr::Bold).unwrap();
                                        t.fg(term::color::BLUE).unwrap();
                                        (write!(t, "{}.", func_name[0])).unwrap();
                                        t.reset().unwrap();
                                        i = 1;

                                        while i < func_name.len() {
                                            (write!(t, "({}", func_name[i])).unwrap();
                                            i += 1;
                                        }
                                    }
                                } else {
                                    (write!(t, "{}", funcs[0])).unwrap();
                                }
                            }
                            None => {
                                (write!(t, "{}", word)).unwrap();
                            }
                        }
                    } else {
                        let func_name : Vec<&str> = word.split('(').collect();

                        if func_name.len() > 1 {
                            t.attr(term::attr::Bold).unwrap();
                            t.fg(term::color::BLUE).unwrap();
                            (write!(t, "{}", func_name[0])).unwrap();
                            t.reset().unwrap();
                            let mut i = 1;

                            while i < func_name.len() {
                                (write!(t, "({}", func_name[i])).unwrap();
                                i += 1;
                            }
                        } else {
                            (write!(t, "{}", word)).unwrap();
                        }
                    }
                }
            }
            it += 1;
            if it < words.len() {
                (write!(t, " ")).unwrap();
            }
        }
        total -= 1;
        if total > 1 {
            (writeln!(t, "")).unwrap();
        }
    }
}

pub fn beautiful_error_printing(splits: &[&str]) {
    let mut i = 1;

    while i < splits.len() {
        print!("{}", splits[i - 1]);
        parse_input(splits[i]);
        i += 2;
        if i < splits.len() {
            println!("");
        }
    }
    if i - 1 < splits.len() {
        print!("{}", splits[i - 1]);
    }
}