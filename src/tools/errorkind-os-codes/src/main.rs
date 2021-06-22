
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem;

use regex::Regex;

struct Os {
    pub name: &'static str,
    pub src: &'static str,
    pub fn_start: &'static str,
    pub scope_retxt: &'static str,
}

fn some_codes(dir: &str, Os { src, fn_start, scope_retxt,.. }: &Os)
              -> HashMap<String, Vec<String>>
{
    let src = format!("{}/{}", dir, src);
    let f = File::open(&src).expect(&format!("failed to open {}", &src));
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
    let vprefix_re = Regex::new(&format!(r#"^{}"#, scope_retxt)).unwrap();

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
            for e in mem::take(&mut current).split('|') {
                let e = e.trim();
                if e.len() == 0 { continue }
                if let Some(m) = vprefix_re.find(e) {
                    let v = &e[m.end()..];
                    out.entry(key.to_owned())
                        .or_insert(vec![])
                        .push(v.to_owned())
                }
            }
        } else {
            panic!("{}:{}: uncategorisable line {:?}", src, lno, &l);
        }
    }
    out
}

static OSLIST: &[Os] = &[
    Os {
        name: "Windows",
        src: "src/sys/windows/mod.rs",
        fn_start: "pub fn decode_error_kind(",
        scope_retxt: r#"c::"#,
    },
];

fn main() {
    let dir = "library/std/";
    let osses = OSLIST.iter().map(|os| (os.name, some_codes(dir, os))).collect::<Vec<_>>();
    eprintln!("{:#?}", &osses);
}
