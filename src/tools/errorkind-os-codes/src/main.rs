
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem;

use regex::Regex;

fn some_codes(src: &str, fn_start: &str, scope_retxt: &str) -> HashMap<String, Vec<String>> {
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

    let ignore_re = Regex::new(r#"^ *(?:use |match| _ *=> | +\}|  } *| *$)"#).unwrap();
    let ent_retxt = format!(r#"{}\w+\b"#, scope_retxt);
    let lhs_retxt = format!(r#"^ *\|? +({}(?: *\| *{})*)"#, &ent_retxt, &ent_retxt);
    let part_re = Regex::new(&format!(r#"({}) *\|? *$"#, &lhs_retxt)).unwrap();
    let full_retxt = format!(r#"({}) *=> *(?:return)? *(?P<k>\w+),? *$"#, &lhs_retxt);
    let full_re = Regex::new(&full_retxt).unwrap();
    eprintln!("lhs_retxt  {}", &lhs_retxt);
    eprintln!("full_retxt {}", &full_retxt);


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
            let key = m.name("k").unwrap().as_str();
dbg!(&current, &key);
            for e in mem::take(&mut current).split('|') {
                let e = e.trim();
                if e.len() == 0 { continue }
                out.entry(key.to_owned())
                    .or_insert(vec![])
                    .push(e.to_owned())
            }
        } else {
            panic!("{}:{}: uncategorisable line {:?}", src, lno, &l);
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
