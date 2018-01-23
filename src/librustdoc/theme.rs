// Copyright 2012-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;

macro_rules! try_false {
    ($e:expr) => ({
        match $e {
            Ok(c) => c,
            Err(e) => {
                eprintln!("rustdoc: got an error: {}", e);
                return false;
            }
        }
    })
}

#[derive(Debug, Clone, Eq)]
pub struct CssPath {
    pub name: String,
    pub children: HashSet<CssPath>,
}

// This PartialEq implementation IS NOT COMMUTATIVE!!!
//
// The order is very important: the second object must have all first's rules.
// However, the first doesn't require to have all second's rules.
impl PartialEq for CssPath {
    fn eq(&self, other: &CssPath) -> bool {
        if self.name != other.name {
            false
        } else {
            for child in &self.children {
                if !other.children.iter().any(|c| child == c) {
                    return false;
                }
            }
            true
        }
    }
}

impl Hash for CssPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        for x in &self.children {
            x.hash(state);
        }
    }
}

impl CssPath {
    fn new(name: String) -> CssPath {
        CssPath {
            name,
            children: HashSet::new(),
        }
    }
}

/// All variants contain the position they occur.
#[derive(Debug, Clone, Copy)]
enum Events {
    StartLineComment(usize),
    StartComment(usize),
    EndComment(usize),
    InBlock(usize),
    OutBlock(usize),
}

impl Events {
    fn get_pos(&self) -> usize {
        match *self {
            Events::StartLineComment(p) |
            Events::StartComment(p) |
            Events::EndComment(p) |
            Events::InBlock(p) |
            Events::OutBlock(p) => p,
        }
    }

    fn is_comment(&self) -> bool {
        match *self {
            Events::StartLineComment(_) |
            Events::StartComment(_) |
            Events::EndComment(_) => true,
            _ => false,
        }
    }
}

fn previous_is_line_comment(events: &[Events]) -> bool {
    if let Some(&Events::StartLineComment(_)) = events.last() {
        true
    } else {
        false
    }
}

fn is_line_comment(pos: usize, v: &[u8], events: &[Events]) -> bool {
    if let Some(&Events::StartComment(_)) = events.last() {
        return false;
    }
    pos + 1 < v.len() && v[pos + 1] == b'/'
}

fn load_css_events(v: &[u8]) -> Vec<Events> {
    let mut pos = 0;
    let mut events = Vec::with_capacity(100);

    while pos < v.len() - 1 {
        match v[pos] {
            b'/' if pos + 1 < v.len() && v[pos + 1] == b'*' => {
                events.push(Events::StartComment(pos));
                pos += 1;
            }
            b'/' if is_line_comment(pos, v, &events) => {
                events.push(Events::StartLineComment(pos));
                pos += 1;
            }
            b'\n' if previous_is_line_comment(&events) => {
                events.push(Events::EndComment(pos));
            }
            b'*' if pos + 1 < v.len() && v[pos + 1] == b'/' => {
                events.push(Events::EndComment(pos + 2));
                pos += 1;
            }
            b'{' if !previous_is_line_comment(&events) => {
                if let Some(&Events::StartComment(_)) = events.last() {
                    pos += 1;
                    continue
                }
                events.push(Events::InBlock(pos + 1));
            }
            b'}' if !previous_is_line_comment(&events) => {
                if let Some(&Events::StartComment(_)) = events.last() {
                    pos += 1;
                    continue
                }
                events.push(Events::OutBlock(pos + 1));
            }
            _ => {}
        }
        pos += 1;
    }
    events
}

fn get_useful_next(events: &[Events], pos: &mut usize) -> Option<Events> {
    while *pos < events.len() {
        if !events[*pos].is_comment() {
            return Some(events[*pos]);
        }
        *pos += 1;
    }
    None
}

fn inner(v: &[u8], events: &[Events], pos: &mut usize) -> HashSet<CssPath> {
    let mut pathes = Vec::with_capacity(50);

    while *pos < events.len() {
        if let Some(Events::OutBlock(_)) = get_useful_next(events, pos) {
            println!("00 => {:?}", events[*pos]);
            *pos += 1;
            break
        }
        println!("a => {:?}", events[*pos]);
        if let Some(Events::InBlock(start_pos)) = get_useful_next(events, pos) {
            println!("aa => {:?}", events[*pos]);
            pathes.push(CssPath::new(::std::str::from_utf8(if *pos > 0 {
                &v[events[*pos - 1].get_pos()..start_pos - 1]
            } else {
                &v[..start_pos]
            }).unwrap_or("").trim().to_owned()));
            *pos += 1;
        }
        println!("b => {:?}", events[*pos]);
        while let Some(Events::InBlock(_)) = get_useful_next(events, pos) {
            println!("bb => {:?}", events[*pos]);
            if let Some(ref mut path) = pathes.last_mut() {
                for entry in inner(v, events, pos).iter() {
                    path.children.insert(entry.clone());
                }
            }
        }
        if *pos < events.len() {
            println!("c => {:?}", events[*pos]);
        }
        if let Some(Events::OutBlock(_)) = get_useful_next(events, pos) {
            *pos += 1;
        }
    }
    pathes.iter().cloned().collect()
}

pub fn load_css_pathes(v: &[u8]) -> CssPath {
    let events = load_css_events(v);
    let mut pos = 0;

    println!("\n======> {:?}", events);
    let mut parent = CssPath::new("parent".to_owned());
    parent.children = inner(v, &events, &mut pos);
    parent
}

pub fn test_theme_against<P: AsRef<Path>>(f: &P, against: &CssPath) -> bool {
    let mut file = try_false!(File::open(f));
    let mut data = Vec::with_capacity(1000);

    try_false!(file.read_to_end(&mut data));
    let pathes = load_css_pathes(&data);
    println!("========= {:?}", pathes);
    println!("========= {:?}", against);
    pathes == *against
}

#[test]
fn test_comments_in_rules() {
    let text = r#"
rule a {}

rule b, c
// a line comment
{}

rule d
// another line comment
e {}

rule f/* a multine

comment*/{}

rule g/* another multine

comment*/h

i {}

rule j/*commeeeeent

you like things like "{}" in there? :)
*/
end {}
"#;
}