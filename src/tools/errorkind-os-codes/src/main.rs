
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

fn some_codes(src: &str, fn_start: &str, scope_retxt: &str) -> HashMap<String, String> {
    let f = File::open(src).expect(&format!("failed to open {}", src));
    let f = BufReader::new(f);
    let mut f = f.lines().zip(1..);

    'fn_start: loop {
        for (l,_lno) in &mut f {
            let l = l.expect("read line");
            if l.starts_with(fn_start) { break 'fn_start }
        }
        panic!("failed to find {:?} in {}", fn_start, src);
    };

    let ignore_re = Regex::new(r#"^ *(?:use |match| _ *=> | +\}|  } *| *$"#).unwrap();
    let ent_retxt = format!(r#"{}\w+\b"#, scope_retxt);
    let lhs_retxt = format!(r#"^ *\|? +({}(?: *\| *{})*)"#, &ent_retxt, &ent_retxt);
    let part_re = Regex::new(&format!(r#"({}) *\|? *$"#, &lhs_retxt)).unwrap();
    let full_re = Regex::new(&format!(r#"({}) *=> *(?:return)? *(\w+),? *$#"#, &lhs_retxt)).unwrap();

    let mut current = String::new();
    let mut out = HashMap::new();
    let more_current = |c: &mut String, m: &regex::Captures| {
        *c += " | ";
        *c += m.get(1).unwrap().as_str();
    };

    for (l, lno) in &mut f {
        let l = l.expect("read line");
        if l.trim_end() == "}" {
            break
        } else if ignore_re.is_match(&l) {
            // ok
        } else if let Some(m) = part_re.captures(&l) {
            more_current(&mut current, &m);
        } else if let Some(m) = full_re.captures(&l) {
            more_current(&mut current, &m);
            let key = m.get(2).unwrap().as_str();
            for e in current.split('|') {
                let e = e.trim();
                if e.len() == 0 { continue }
                if let Some(_) = out.insert(key.to_owned(), e.to_owned()) {
                    panic!("duplicate mapping in {} for {:?}", src, &e);
                }
            }
        } else {
            panic!("{}:{}: uncategorisable line", src, lno);
        }
    }
    out
}

fn main() {
    let k = some_codes("library/std/src/sys/windows/mod.rs",
                       "pub fn decode_error_kind(",
                       r#"c::"#);
    eprintln!("{:#?}", k);
}
