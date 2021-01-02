use rustc_data_structures::fx::FxHashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use rustc_errors::Handler;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Eq)]
crate struct CssPath {
    crate name: String,
    crate children: FxHashSet<CssPath>,
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
        CssPath { name, children: FxHashSet::default() }
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
            Events::StartLineComment(p)
            | Events::StartComment(p)
            | Events::EndComment(p)
            | Events::InBlock(p)
            | Events::OutBlock(p) => p,
        }
    }

    fn is_comment(&self) -> bool {
        matches!(self, Events::StartLineComment(_) | Events::StartComment(_) | Events::EndComment(_))
    }
}

fn previous_is_line_comment(events: &[Events]) -> bool {
    matches!(events.last(), Some(&Events::StartLineComment(_)))
}

fn is_line_comment(pos: usize, v: &[u8], events: &[Events]) -> bool {
    if let Some(&Events::StartComment(_)) = events.last() {
        return false;
    }
    v[pos + 1] == b'/'
}

fn load_css_events(v: &[u8]) -> Vec<Events> {
    let mut pos = 0;
    let mut events = Vec::with_capacity(100);

    while pos + 1 < v.len() {
        match v[pos] {
            b'/' if v[pos + 1] == b'*' => {
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
            b'*' if v[pos + 1] == b'/' => {
                events.push(Events::EndComment(pos + 2));
                pos += 1;
            }
            b'{' if !previous_is_line_comment(&events) => {
                if let Some(&Events::StartComment(_)) = events.last() {
                    pos += 1;
                    continue;
                }
                events.push(Events::InBlock(pos + 1));
            }
            b'}' if !previous_is_line_comment(&events) => {
                if let Some(&Events::StartComment(_)) = events.last() {
                    pos += 1;
                    continue;
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

fn get_previous_positions(events: &[Events], mut pos: usize) -> Vec<usize> {
    let mut ret = Vec::with_capacity(3);

    ret.push(events[pos].get_pos());
    if pos > 0 {
        pos -= 1;
    }
    loop {
        if pos < 1 || !events[pos].is_comment() {
            let x = events[pos].get_pos();
            if *ret.last().unwrap() != x {
                ret.push(x);
            } else {
                ret.push(0);
            }
            break;
        }
        ret.push(events[pos].get_pos());
        pos -= 1;
    }
    if ret.len() & 1 != 0 && events[pos].is_comment() {
        ret.push(0);
    }
    ret.iter().rev().cloned().collect()
}

fn build_rule(v: &[u8], positions: &[usize]) -> String {
    minifier::css::minify(
        &positions
            .chunks(2)
            .map(|x| ::std::str::from_utf8(&v[x[0]..x[1]]).unwrap_or(""))
            .collect::<String>()
            .trim()
            .replace("\n", " ")
            .replace("/", "")
            .replace("\t", " ")
            .replace("{", "")
            .replace("}", "")
            .split(' ')
            .filter(|s| !s.is_empty())
            .collect::<Vec<&str>>()
            .join(" "),
    )
    .unwrap_or_else(|_| String::new())
}

fn inner(v: &[u8], events: &[Events], pos: &mut usize) -> FxHashSet<CssPath> {
    let mut paths = Vec::with_capacity(50);

    while *pos < events.len() {
        if let Some(Events::OutBlock(_)) = get_useful_next(events, pos) {
            *pos += 1;
            break;
        }
        if let Some(Events::InBlock(_)) = get_useful_next(events, pos) {
            paths.push(CssPath::new(build_rule(v, &get_previous_positions(events, *pos))));
            *pos += 1;
        }
        while let Some(Events::InBlock(_)) = get_useful_next(events, pos) {
            if let Some(ref mut path) = paths.last_mut() {
                for entry in inner(v, events, pos).iter() {
                    path.children.insert(entry.clone());
                }
            }
        }
        if let Some(Events::OutBlock(_)) = get_useful_next(events, pos) {
            *pos += 1;
        }
    }
    paths.iter().cloned().collect()
}

crate fn load_css_paths(v: &[u8]) -> CssPath {
    let events = load_css_events(v);
    let mut pos = 0;

    let mut parent = CssPath::new("parent".to_owned());
    parent.children = inner(v, &events, &mut pos);
    parent
}

crate fn get_differences(against: &CssPath, other: &CssPath, v: &mut Vec<String>) {
    if against.name == other.name {
        for child in &against.children {
            let mut found = false;
            let mut found_working = false;
            let mut tmp = Vec::new();

            for other_child in &other.children {
                if child.name == other_child.name {
                    if child != other_child {
                        get_differences(child, other_child, &mut tmp);
                    } else {
                        found_working = true;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                v.push(format!("  Missing \"{}\" rule", child.name));
            } else if !found_working {
                v.extend(tmp.iter().cloned());
            }
        }
    }
}

crate fn test_theme_against<P: AsRef<Path>>(
    f: &P,
    against: &CssPath,
    diag: &Handler,
) -> (bool, Vec<String>) {
    let data = match fs::read(f) {
        Ok(c) => c,
        Err(e) => {
            diag.struct_err(&e.to_string()).emit();
            return (false, vec![]);
        }
    };

    let paths = load_css_paths(&data);
    let mut ret = vec![];
    get_differences(against, &paths, &mut ret);
    (true, ret)
}
